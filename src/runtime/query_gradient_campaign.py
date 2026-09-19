"""Sequential, fail-closed QCGS campaign with committed inputs and resumable cells."""
from __future__ import annotations

import argparse
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch
import yaml

from .query_gradient_registry import CELLS, STAGES, build_registry, digest, file_sha, write_json, validate_registry
from .artifact_provenance import verify_cifar100c_provenance, verify_cached_clip_checkpoint
from .cifar100c_huggingface import CIFAR100C_NPY_MD5
from evaluation.query_gradient_utility import (ANALYSIS, analyze, audit_rows, freeze_bins, read_jsonl,
                                               require, write_jsonl, write_report)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'cfg/research/query-gradient-diagnostic'
SPLIT = ROOT/'cfg/research/open-set-cifar100-split-v1.json'
PROTOCOL = ROOT/'docs/research/query-conditioned-gradient-selection.md'


def profile(rescue):
    if rescue:
        from evaluation import query_gradient_multiview as analysis
        return (ROOT/'cfg/research/query-gradient-multiview',
                ROOT/'docs/research/qcgs-multiview-rescue-protocol.md',
                'MultiViewQueryGradientUtilityProbe', analysis)
    from evaluation import query_gradient_utility as analysis
    return CONFIG, PROTOCOL, 'QueryGradientUtilityProbe', analysis


class InterruptedCampaign(RuntimeError):
    """Resource interruption or missing quota: not scientific evidence of failure."""


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args],cwd=cwd,text=True).strip()


def source_identity(*, clean, rescue=False):
    CONFIG, PROTOCOL, _, _ = profile(rescue)
    revision = git('rev-parse','HEAD')
    spec = json.loads((CONFIG/'protocol.json').read_text())
    subprocess.run(['git','merge-base','--is-ancestor',spec['base_revision'],revision],cwd=ROOT,check=True)
    status = git('status','--porcelain','--untracked-files=normal')
    if clean and status:
        raise ValueError('scientific execution requires a clean committed source checkout')
    return {'revision':revision,'base_revision':spec['base_revision'],'tree':git('rev-parse','HEAD^{tree}'),
            'dirty':bool(status),'protocol_sha256':file_sha(PROTOCOL),'spec_sha256':file_sha(CONFIG/'protocol.json'),
            'exclusions_sha256':file_sha(CONFIG/'historical-exclusions.json'),'split_sha256':file_sha(SPLIT)}


def environment():
    cuda = torch.cuda.is_available()
    driver = subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip() if cuda else None
    return {'python':platform.python_version(), 'system':platform.system(),
            'packages':{p:metadata.version(p) for p in ('torch','torchvision','numpy','PyYAML','Pillow','tqdm','clip','pytest')},
            'installed_distributions':dict(sorted((d.metadata['Name'],d.version) for d in metadata.distributions() if d.metadata['Name'])),
            'cuda_available':cuda,'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
            'driver':driver, 'cublas_workspace_config':':4096:8','deterministic_algorithms':True,'allow_tf32':False}


def run_tests(destination, *, rescue=False):
    destination.mkdir(parents=True,exist_ok=True)
    commands = [[sys.executable,'-m','pytest','-q','tests/test_oracle_support_utility.py','tests/test_query_gradient_selection.py'],
                [sys.executable,'-m','pytest','-q']]
    if rescue:
        commands[0].extend(['tests/test_query_gradient_multiview.py', 'tests/test_multiview_analysis.py'])
    env = {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1','PYTHONDONTWRITEBYTECODE':'1'}
    for i,command in enumerate(commands):
        print('CPU verification:', ' '.join(command),flush=True)
        with (destination/f'cpu-tests-{i}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    value = {'passed':True,'commands':commands,'cuda_hidden_for_tests':True,
             'logs':{f'cpu-tests-{i}.log':file_sha(destination/f'cpu-tests-{i}.log') for i in range(2)}}
    write_json(destination/'cpu-tests.json',value)
    return value


def checkpoint(evidence):
    spec = util.spec_from_file_location('qcgs_checkpoint',ROOT/'notebooks/kaggle/full-run-checkpoint.py')
    module = util.module_from_spec(spec);spec.loader.exec_module(module)
    result = module.checkpoint(evidence)
    print('Checkpoint:',result,flush=True)
    return result


def freeze_file(path, value):
    if path.exists():
        require(json.loads(path.read_text()) == value, f'frozen input changed: {path.name}')
    else:
        write_json(path,value)


def ledger_commit(locks, message):
    if not (locks/'.git').exists():
        subprocess.run(['git','init','-q',str(locks)],check=True)
    git('add','--all',cwd=locks)
    if git('status','--porcelain',cwd=locks):
        git('-c','user.name=QCGS Evidence','-c','user.email=qcgs-evidence@localhost',
            '-c','commit.gpgsign=false','commit','-qm',message,cwd=locks)
    return git('rev-parse','HEAD',cwd=locks)


def verify_ledger(locks):
    require((locks/'.git').is_dir(),'missing evidence input Git ledger')
    require(not git('status','--porcelain',cwd=locks),'frozen inputs differ from committed ledger')
    subprocess.run(['git','fsck','--no-reflogs'],cwd=locks,check=True,stdout=subprocess.DEVNULL)


def config_for(mode, cell, fingerprint, revision, registry=None, *, rescue=False):
    CONFIG, _, _, analysis = profile(rescue)
    cfg = {key:json.loads((CONFIG/'protocol.json').read_text())[key]
           for key in ('max_capacity','topk','beta','optimizer','lr','candidate_m')}
    cfg.update(probe_queries=STAGES.get(mode,2),probe_mode='exhaustive' if mode=='stage-a' else 'screened',
               oracle_label_source='evaluator_known_label',qcgs_mode=mode,ood_cell=cell,
               dataset_fingerprint=fingerprint,source_revision=revision)
    if rescue:
        cfg['view_spec'] = analysis.VIEW_SPEC
    if registry is not None:
        cfg.update(registry_path=str(registry),registry_file_sha256=file_sha(registry))
    return cfg


def make_job(locks, runs, data_root, mode, cell, cap, fingerprint, revision, registry=None, *, rescue=False):
    _, _, method, _ = profile(rescue)
    name = f'{mode}-ood-{cell}-n{cap}'
    config = config_for(mode,cell,fingerprint,revision,registry,rescue=rescue)
    order_path = locks/'parameter-order.json'
    if mode != 'smoke' and order_path.exists():
        state = json.loads(order_path.read_text())
        config['parameter_order_sha256'] = state['parameter_order_sha256']
        if rescue:
            config['view_state_sha256'] = state['view_state']['sha256']
    config_root = locks/'configs'/name
    path = config_root/'CIFAR100C'/f'{method}.yaml'
    encoded = yaml.safe_dump(config,sort_keys=True)
    if path.exists():
        require(path.read_text()==encoded,'frozen YAML changed')
    else:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(encoded)
    command = [sys.executable,str(ROOT/'src/main.py'),'--dataset','CIFAR100C','--data_root',str(data_root),
               '--artifact-provenance','fast','--open_set','--ood_ratio',cell,
               '--known_class_split','open-set-cifar100-split-v1',
               '--known-class-split-path',str(SPLIT),'--known-class-split-sha256',file_sha(SPLIT),
               '--model','clip_vitbase16','--tta_algo',method,'--batch_size','100',
               '--device','cuda','--config',str(config_root),'--config-lock-path',str(path),
               '--config-lock-sha256',file_sha(path),'--stream_mode','block','--stream_block_size','64',
               '--seed','0','--stream_seed','0','--num_workers','0','--open_set_per_domain_source_budget','400',
               '--max_eval_samples',str(cap),'--run_id',name,'--evidence_dir',str(runs),'--save_to',str(runs/'runs.csv')]
    return {'name':name,'mode':mode,'cell':cell,'cap':cap,'command':command,
            'config_sha256':digest(config),'config_file_sha256':file_sha(path),'config_file':str(path),
            **({'diagnostic':'qcgs-multiview'} if rescue else {})}


def validate_run(job, runs, source, artifacts, registry=None):
    rescue = job.get('diagnostic') == 'qcgs-multiview'
    _, _, _, analysis = profile(rescue)
    folder = runs/job['name']
    manifest = json.loads((folder/'manifest.json').read_text())
    require(manifest['git']['commit']==source['revision'] and not manifest['git']['dirty'], 'run source revision/cleanliness mismatch')
    require(digest(manifest['config'])==job['config_sha256'], 'manifest config mismatch')
    state=json.loads((folder/'qcgs-state.json').read_text())
    require(digest(state['parameter_order'])==state['parameter_order_sha256'],'parameter ordering hash mismatch')
    require(manifest['config'].get('parameter_order_sha256',state['parameter_order_sha256'])==state['parameter_order_sha256'],'parameter ordering changed')
    require(manifest['artifacts']['dataset']['root_digest']==artifacts['dataset']['root_digest'], 'dataset identity changed')
    require(manifest['artifacts']['model']['actual_sha256']==artifacts['model']['actual_sha256'], 'model identity changed')
    require(manifest['args']['known_class_split_sha256']==source['split_sha256'], 'split identity changed')
    require(manifest['args']['max_eval_samples']==job['cap'] and manifest['args']['device']=='cuda', 'run cap/device mismatch')
    if rescue:
        require(state['view_state']['view_spec']==analysis.VIEW_SPEC, 'run view specification changed')
        require(manifest['config'].get('view_state_sha256',state['view_state']['sha256'])==state['view_state']['sha256'], 'view/preprocessing lock changed')
    scan = read_jsonl(folder/'qcgs-scan.jsonl')
    trace = read_jsonl(folder/'trace.jsonl')
    require(len(scan)==job['cap'] and len(trace)==job['cap'], 'incomplete stream trajectory')
    for i,(a,b) in enumerate(zip(scan,trace)):
        require(a['timestep']==b['timestep']==i and a['sample_idx']==b['sample_idx']
                and a['domain']==b['ground_truth_domain'] and a['is_ood']==b['is_ood'], 'scan/trace identity mismatch')
    rows = read_jsonl(folder/'qcgs-queries.jsonl')
    if job['mode']=='scan':
        require(not rows,'preflight scan must not score queries')
    else:
        require(analysis.audit_rows(rows,job['mode'],job['cell'],registry,job['config_sha256']), 'incomplete scored-query quota')
        for row in rows:
            require(row['parameter_order_sha256']==state['parameter_order_sha256'],'query parameter ordering mismatch')
            if rescue:
                require(row['view_state']==state['view_state'], 'query view/preprocessing provenance mismatch')
            observation = trace[row['timestep']]
            require(row['known_label']==observation['known_label_or_minus_one'] and row['ramen']['prediction']==observation['prediction'], 'evaluator label/baseline trajectory mismatch')
    if registry is not None:
        require(scan==registry['cells'][job['cell']]['stream'][:job['cap']], 'registry replay mismatch')
    return rows, scan


def artifact_hashes(folder):
    return {str(p.relative_to(folder)):file_sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name!='completed.json' and not p.name.endswith('.part')}


def run_job(job, evidence, source, artifacts, timeout, registry=None):
    require(source_identity(clean=True,rescue=job.get('diagnostic')=='qcgs-multiview')==source,'source changed after preflight')
    runs = evidence/'runs';runs.mkdir(exist_ok=True)
    folder = runs/job['name']; receipt = folder/'completed.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        require(saved['job']==job and saved['files']==artifact_hashes(folder),'completed cell changed after checkpoint')
        verify_ledger(evidence/'locks')
        return validate_run(job,runs,source,artifacts,registry)
    if folder.exists():
        # Whole-cell restart preserves interrupted evidence and RNG/trajectory semantics.
        abandoned = evidence/'interrupted';abandoned.mkdir(exist_ok=True)
        folder.rename(abandoned/(job['name']+'-'+str(time.time_ns())))
    log_path = runs/(job['name']+'.log')
    if log_path.exists():
        abandoned = evidence/'interrupted';abandoned.mkdir(exist_ok=True)
        log_path.rename(abandoned/(log_path.name+'-'+str(time.time_ns())))
    print('RUN',job['name'],flush=True)
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0','PYTHONDONTWRITEBYTECODE':'1'}
    started=time.monotonic()
    with log_path.open('w') as log:
        process=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    returncode=process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f'Running {job["name"]}: {time.monotonic()-started:.0f}s; {log_path}',flush=True)
                    checkpoint(evidence)
                    if time.monotonic()-started > timeout:
                        raise InterruptedCampaign(f'timeout: {job["name"]}; partial cell retained, no later stage started')
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill();process.wait()
            raise
    if returncode:
        tail=log_path.read_text()[-12000:]
        if returncode in (-9,-15) or 'CUDA out of memory' in tail:
            raise InterruptedCampaign(f'resource interruption: {job["name"]}; inspect {log_path}')
        raise ValueError(f'child failed ({returncode}): {job["name"]}; {tail[-3000:]}')
    require(source_identity(clean=True,rescue=job.get('diagnostic')=='qcgs-multiview')==source,'source changed during run')
    result=validate_run(job,runs,source,artifacts,registry)
    write_json(receipt,{'job':job,'files':artifact_hashes(folder),'input_commit':git('rev-parse','HEAD',cwd=evidence/'locks')})
    checkpoint(evidence)
    return result


def audit_campaign(evidence, *, rescue=False):
    _, _, method, analysis = profile(rescue)
    ANALYSIS, audit_rows, analyze, write_report = analysis.ANALYSIS, analysis.audit_rows, analysis.analyze, analysis.write_report
    locks=evidence/'locks';verify_ledger(locks)
    launch=json.loads((locks/'launch.json').read_text())
    registry=json.loads((locks/'registry.json').read_text());validate_registry(registry)
    bins=json.loads((locks/'bins.json').read_text()) if (locks/'bins.json').exists() else None
    gate=json.loads((locks/'stage-a-gate.json').read_text()) if (locks/'stage-a-gate.json').exists() else None
    frozen=json.loads((locks/'preflight.json').read_text())
    require(launch['preflight_sha256']==file_sha(locks/'preflight.json'), 'preflight content lock mismatch')
    require(launch['registry_file_sha256']==file_sha(locks/'registry.json')
            and launch['registry_sha256']==registry['sha256'], 'registry content lock mismatch')
    require(launch['analysis_sha256']==digest(ANALYSIS) and frozen['analysis']==ANALYSIS,
            'analysis specification changed; use the pinned diagnostic implementation')
    require(launch['source_revision']==frozen['source']['revision']==registry['source_revision'], 'launch source lock mismatch')
    require(len(launch['jobs'])==4 and {(j['mode'],j['cell']) for j in launch['jobs']}=={(s,c) for s in STAGES for c in CELLS},
            'launch must contain each registered stage/cell exactly once')
    cells={s:{c:[] for c in CELLS} for s in STAGES}
    failure=evidence/'integrity-failure.json'
    errors=[json.loads(failure.read_text())['reason']] if failure.exists() else []
    for job in launch['jobs']:
        folder=evidence/'runs'/job['name']
        try:
            config_path=locks/'configs'/job['name']/'CIFAR100C'/f'{method}.yaml'
            require((job.get('diagnostic')=='qcgs-multiview')==rescue,'mixed diagnostic jobs')
            if rescue and job['mode']=='stage-b' and folder.exists():
                require(gate is not None and gate['decision']=='GO_CONFIRM','Stage B started without GO_CONFIRM')
            require(file_sha(config_path)==job['config_file_sha256'],'locked config file changed')
            if (folder/'completed.json').exists():
                receipt=json.loads((folder/'completed.json').read_text())
                require(receipt['job']==job and receipt['files']==artifact_hashes(folder),'completed artifact hashes mismatch')
                # The frozen launch must precede execution in the evidence ledger.
                locked_launch=git('show',receipt['input_commit']+':launch.json',cwd=locks)
                require(json.loads(locked_launch)==launch,'launch not committed before scored execution')
                if job['mode']=='stage-b':
                    if rescue:
                        require(json.loads(git('show',receipt['input_commit']+':stage-a-gate.json',cwd=locks))==gate,'GO_CONFIRM not committed before Stage B')
                    else:
                        require(json.loads(git('show',receipt['input_commit']+':bins.json',cwd=locks))==bins,'bins not committed before Stage B')
                rows,_=validate_run(job,evidence/'runs',frozen['source'],frozen['artifacts'],registry)
                cells[job['mode']][job['cell']]=rows
            elif (folder/'qcgs-queries.jsonl').exists():
                # Still audit finite partial rows; never promote them to completed evidence.
                rows=read_jsonl(folder/'qcgs-queries.jsonl')
                audit_rows(rows,job['mode'],job['cell'],registry,job['config_sha256'])
        except (ValueError,KeyError,TypeError,IndexError,OSError,subprocess.CalledProcessError) as exc:
            errors.append(f'{job["name"]}: {exc}')
    if rescue and gate is not None:
        try:
            require(gate==analysis.stage_a_gate(cells['stage-a'],registry),'committed Stage A gate disagrees with raw evidence')
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            errors.append(str(exc))
    summary=analyze(cells['stage-a'],cells['stage-b'],registry,bins,integrity_errors=errors)
    write_jsonl(evidence/'queries.jsonl',[r for stage in STAGES for cell in CELLS for r in cells[stage][cell]])
    write_report(evidence,summary)
    return summary


def execute(args):
    rescue = args.rescue
    CONFIG, _, _, analysis = profile(rescue)
    ANALYSIS = analysis.ANALYSIS
    args.data_root = args.data_root.expanduser().resolve()
    evidence=args.evidence_dir.resolve();locks=evidence/'locks'
    source=source_identity(clean=True,rescue=rescue)
    require(not (evidence/'integrity-failure.json').exists(),'invalid campaign is sealed; repair and use a new evidence directory')
    spec=json.loads((CONFIG/'protocol.json').read_text())
    if evidence.exists() and any(p.name != 'runtime' for p in evidence.iterdir()) and not args.resume:
        raise FileExistsError('use a fresh evidence directory or explicitly --resume the identical campaign')
    locks.mkdir(parents=True,exist_ok=True)
    if (locks/'preflight.json').exists():
        verify_ledger(locks)
        frozen=json.loads((locks/'preflight.json').read_text())
        require(frozen['source']==source,'source changed; never resume after semantic changes')
        require(frozen['environment']==environment(),'environment changed; start fresh evidence')
    else:
        tests=run_tests(locks,rescue=rescue)
        if not torch.cuda.is_available():
            raise InterruptedCampaign('CUDA unavailable; no smoke or scientific queries were run')
        artifacts={'dataset':verify_cifar100c_provenance(args.data_root/'corruption/CIFAR-100-C',exact=True),
                   'model':verify_cached_clip_checkpoint('clip_vitbase16',Path.home()/'.cache/clip')}
        # The historical base-index exclusions require original NPY order, independently of transport/root README.
        from .artifact_provenance import checksum_regular_file
        for name,expected in CIFAR100C_NPY_MD5.items():
            require(checksum_regular_file(args.data_root/'corruption/CIFAR-100-C'/name,'md5')['checksum']==expected,'historical index binding checksum mismatch')
        frozen={'source':source,'environment':environment(),'artifacts':artifacts,'cpu_tests':tests,
                'analysis':ANALYSIS,'protocol':spec,'data_root':str(args.data_root.resolve()),
                'parity':{'rtol':0,'atol':0},'timeout_seconds':args.timeout_seconds or spec['timeout_seconds_per_cell']}
        freeze_file(locks/'preflight.json',frozen)
        freeze_file(locks/'historical-exclusions.json',json.loads((CONFIG/'historical-exclusions.json').read_text()))
        ledger_commit(locks,'chore: freeze QCGS preflight and smoke prerequisites')
    require(str(args.data_root.resolve())==frozen['data_root'],'data path changed on resume')
    if args.timeout_seconds is not None:
        require(args.timeout_seconds==frozen['timeout_seconds'],'timeout must be locked before execution')
    artifacts=frozen['artifacts'];fingerprint=artifacts['dataset']['root_digest'];revision=source['revision']
    # Rehash content on resume too; fast child manifests are backed by this exact check.
    if args.resume:
        actual=verify_cifar100c_provenance(args.data_root/'corruption/CIFAR-100-C',exact=True)
        require(actual['root_digest']==fingerprint,'dataset changed on resume')
        require(verify_cached_clip_checkpoint('clip_vitbase16',Path.home()/'.cache/clip')['actual_sha256']==artifacts['model']['actual_sha256'],'model changed on resume')
    if not (locks/'registry.json').exists():
        excluded=json.loads((locks/'historical-exclusions.json').read_text())
        smoke_ids=set(excluded['base_image_indices']);smoke_sources=[]
        parameter_order = None
        for cell in CELLS:
            job=make_job(locks,evidence/'runs',args.data_root,'smoke',cell,600,fingerprint,revision,rescue=rescue)
            freeze_file(locks/(job['name']+'.json'),job);ledger_commit(locks,'chore: freeze CUDA smoke launch')
            rows,_=run_job(job,evidence,source,artifacts,frozen['timeout_seconds'])
            observed_order=json.loads((evidence/'runs'/job['name']/'qcgs-state.json').read_text())
            if parameter_order is not None:
                require(parameter_order==observed_order,'smoke parameter ordering differs between cells')
            parameter_order=observed_order
            smoke_ids.update(r['sample_idx'] for r in rows)
            smoke_sources.append({'cell':cell,'queries_sha256':file_sha(evidence/'runs'/job['name']/'qcgs-queries.jsonl'),
                                  'base_image_indices':[r['sample_idx'] for r in rows]})
        freeze_file(locks/'parameter-order.json',parameter_order)
        ledger_commit(locks,'chore: freeze real-model adapted parameter order')
        excluded={**excluded,'base_image_indices':sorted(smoke_ids),'smoke':smoke_sources}
        cap=spec['initial_scan_samples']
        while True:
            scans={}
            for cell in CELLS:
                job=make_job(locks,evidence/'runs',args.data_root,'scan',cell,cap,fingerprint,revision,rescue=rescue)
                freeze_file(locks/(job['name']+'.json'),job);ledger_commit(locks,'chore: freeze outcome-blind query scan')
                _,scans[cell]=run_job(job,evidence,source,artifacts,frozen['timeout_seconds'])
            try:
                registry=build_registry(scans,fingerprint,excluded,source_revision=revision,
                                        provenance={'preflight_sha256':file_sha(locks/'preflight.json'),'scan_cap':cap,
                                                    'extension_reason':'new eligible image quotas only; no scored stage outcomes consulted'},
                                        namespace=spec.get('registry_namespace','qcgs-label-free-v1'))
                break
            except ValueError as exc:
                if not str(exc).startswith('insufficient eligible unseen queries'):
                    raise
                if cap==spec['maximum_scan_samples']:
                    raise InterruptedCampaign(str(exc)) from exc
                cap=min(cap*2,spec['maximum_scan_samples'])
        freeze_file(locks/'registry.json',registry)
        jobs=[make_job(locks,evidence/'runs',args.data_root,stage,cell,registry['cells'][cell]['max_eval_samples'],fingerprint,revision,locks/'registry.json',rescue=rescue)
              for stage in STAGES for cell in CELLS]
        freeze_file(locks/'launch.json',{'jobs':jobs,'source_revision':revision,'registry_sha256':registry['sha256'],
                    'registry_file_sha256':file_sha(locks/'registry.json'),'preflight_sha256':file_sha(locks/'preflight.json'),
                    'analysis_sha256':digest(ANALYSIS)})
        ledger_commit(locks,'chore: lock QCGS queries configs and launch before Stage A')
        checkpoint(evidence)
    verify_ledger(locks)
    registry=json.loads((locks/'registry.json').read_text());validate_registry(registry)
    launch=json.loads((locks/'launch.json').read_text())
    require(launch['registry_file_sha256']==file_sha(locks/'registry.json'),'registry lock changed')
    stage_a={}
    for stage in STAGES:
        for job in (j for j in launch['jobs'] if j['mode']==stage):
            rows,_=run_job(job,evidence,source,artifacts,frozen['timeout_seconds'],registry)
            if stage=='stage-a': stage_a[job['cell']]=rows
        if stage=='stage-a':
            if rescue:
                gate=analysis.stage_a_gate(stage_a,registry)
                freeze_file(locks/'stage-a-gate.json',gate)
                ledger_commit(locks,'chore: commit audited Stage A gate before confirmation')
                checkpoint(evidence)
                if gate['decision']!='GO_CONFIRM':
                    print('Stage A:',gate['decision'],'— Stage B not started',flush=True)
                    break
            else:
                freeze_file(locks/'bins.json',freeze_bins(stage_a))
                ledger_commit(locks,'chore: freeze Stage A analysis bins before Stage B')
                checkpoint(evidence)
    summary=audit_campaign(evidence,rescue=rescue)
    checkpoint(evidence)
    print('Decision:',summary['decision'],flush=True)


def main(argv=None, *, rescue=False):
    _, _, _, analysis = profile(rescue)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('/tmp/nb-ramen-qcgs-data'))
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute',action='store_true');mode.add_argument('--preflight-only',action='store_true');mode.add_argument('--audit',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--timeout-seconds',type=int)
    args=parser.parse_args(argv)
    args.rescue=rescue
    if args.timeout_seconds is not None and args.timeout_seconds<=0: parser.error('timeout must be positive')
    if args.audit:
        print(json.dumps(audit_campaign(args.evidence_dir,rescue=rescue),indent=2));return
    if args.preflight_only:
        args.evidence_dir.mkdir(parents=True,exist_ok=True)
        tests=run_tests(args.evidence_dir,rescue=rescue)
        write_json(args.evidence_dir/'preflight.json',{'source':source_identity(clean=False,rescue=rescue),'cpu_tests':tests,
                   'environment':environment(),'scientific_queries':0,'cuda_smoke_run':False})
        print('CPU preflight passed. CUDA smoke and stages remain unrun.',flush=True);return
    try:
        execute(args)
    except (InterruptedCampaign,KeyboardInterrupt) as exc:
        if args.evidence_dir.exists():
            write_json(args.evidence_dir/'interruption.json',{'decision':'INCONCLUSIVE','reason':str(exc) or 'interrupted'})
            if (args.evidence_dir/'locks/launch.json').exists(): audit_campaign(args.evidence_dir,rescue=rescue)
            checkpoint(args.evidence_dir)
        raise SystemExit(str(exc) or 'Interrupted; checkpoint retained') from exc
    except (ValueError,subprocess.CalledProcessError) as exc:
        if args.evidence_dir.exists():
            write_json(args.evidence_dir/'integrity-failure.json',{'decision':'INVALID','reason':str(exc)})
            analysis.write_report(args.evidence_dir,{'schema_version':2 if rescue else 1,'decision':'INVALID','complete':False,'integrity_errors':[str(exc)],'stage_a':{},'stage_b':{}})
            checkpoint(args.evidence_dir)
        raise


if __name__=='__main__': main()
