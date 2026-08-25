import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import os
import numpy as np
import random
import yaml
import csv
import json
import re
import time
import statistics
from array import array
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from models.get_pretrained_model import get_pretrained_model
from datasets import get_dataset_class
from datasets.open_set import OpenSetCIFAR100C, OpenSetDomainNet
from methods import get_method_class
from streams import build_open_set_stream, build_single_domain_stream, build_stream, truncate_stream
from streams.legacy import build_legacy_torch_iid_stream
from evaluation import (
    JsonlTraceWriter,
    SUMMARY_SCHEMA_VERSION,
    compare_trace_negative_adaptation,
    domain_shift_recovery_times,
    id_accuracy,
    open_set_metrics,
    routing_diagnostics,
    write_run_manifest,
    write_summary,
    verify_reference_trace_stream_fingerprint,
)
from runtime import DeviceMemoryTracker, collect_hardware_evidence
from runtime.artifact_provenance import (
    ProvenanceError,
    verify_cached_clip_checkpoint,
    verify_cifar100c_provenance,
    verify_domainnet_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
OPEN_SET_DATASET_CLASSES = {
    ('CIFAR100C', 'open-set-cifar100-split-v1'): OpenSetCIFAR100C,
    ('DomainNet', 'open-set-domainnet-name-rank-v1'): OpenSetDomainNet,
}
OPEN_SET_SPLIT_FILENAMES = {
    ('CIFAR100C', 'open-set-cifar100-split-v1'): 'open-set-cifar100-split-v1.json',
    ('DomainNet', 'open-set-domainnet-name-rank-v1'): 'open-set-domainnet-split-v1.json',
}


def _open_set_dataset_class(dataset_name, split_name):
    """Return the evaluator wrapper for one supported versioned protocol."""
    dataset_class = OPEN_SET_DATASET_CLASSES.get((dataset_name, split_name))
    if dataset_class is None:
        supported = ', '.join(
            f'{dataset}/{split}' for dataset, split in sorted(OPEN_SET_DATASET_CLASSES)
        )
        raise ValueError(f'unsupported open-set dataset/split; expected one of: {supported}')
    return dataset_class


def _artifact_provenance(args):
    """Return explicit model/dataset evidence before loading either artifact."""
    if args.artifact_provenance == 'off':
        unavailable = {'status': 'unavailable', 'reason': 'artifact provenance disabled by --artifact-provenance off'}
        return {'status': 'unavailable', 'mode': 'off', 'reason': unavailable['reason'],
                'model': dict(unavailable), 'dataset': dict(unavailable)}
    dataset_verifiers = {
        'CIFAR100C': verify_cifar100c_provenance,
        'DomainNet': verify_domainnet_provenance,
    }
    verifier = dataset_verifiers.get(args.dataset)
    if verifier is None:
        raise ProvenanceError(
            '--artifact-provenance fast/exact supports only CIFAR100C and DomainNet'
        )
    dataset_root = (
        Path(args.data_root) / 'corruption' / 'CIFAR-100-C'
        if args.dataset == 'CIFAR100C'
        else Path(args.data_root) / 'domainbed' / 'domain_net'
    )
    # OpenAI CLIP's documented default cache location is used by clip.load;
    # checking it here prevents a later download from changing the run inputs.
    model = verify_cached_clip_checkpoint(args.model, Path.home() / '.cache' / 'clip')
    dataset = verifier(dataset_root, exact=args.artifact_provenance == 'exact')
    return {
        'status': 'verified', 'mode': args.artifact_provenance,
        'model': {'status': 'verified', **model},
        'dataset': {'status': 'verified', **dataset},
    }


def _revalidate_artifact_provenance(args, expected):
    """Reject artifacts changed between preflight and loader construction."""
    if args.artifact_provenance == 'off':
        return expected
    current = _artifact_provenance(args)
    if current != expected:
        raise ProvenanceError('artifact provenance changed while model or dataset was loading')
    return current


def _parse_domain_weights(raw):
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(',')]
    if not values or any(not value for value in values):
        raise argparse.ArgumentTypeError('domain weights must be comma-separated positive numbers')
    try:
        weights = [float(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('domain weights must be comma-separated positive numbers') from exc
    if any(value <= 0 for value in weights):
        raise argparse.ArgumentTypeError('domain weights must be positive')
    return weights


def _sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        mps_backend = getattr(getattr(torch, 'backends', None), 'mps', None)
        is_available = getattr(mps_backend, 'is_available', None)
        if callable(is_available) and is_available() and hasattr(torch, 'mps'):
            torch.mps.synchronize()


def _prefix_segments(segments, retained_sample_count):
    """Clip legacy single-domain reset segments to a retained stream prefix."""
    return tuple(
        (start, min(stop, retained_sample_count))
        for start, stop in segments
        if start < retained_sample_count
    )


def _method_diagnostics(tta_model, batch_size):
    diagnostics = {}
    getter = getattr(tta_model, 'get_diagnostics', None)
    if callable(getter):
        diagnostics = getter() or {}

    if 'memory_size' not in diagnostics and hasattr(tta_model, 'cache'):
        diagnostics['memory_size'] = sum(cache.size for cache in tta_model.cache)

    def expand(name, default=None):
        value = diagnostics.get(name, default)
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(f'diagnostic {name!r} must have one value per sample')
            return list(value)
        return [value] * batch_size

    return {
        'inferred_context': expand('inferred_context'),
        'memory_size': expand('memory_size', 0),
        'num_active_contexts': expand('num_active_contexts'),
        # Method-retained memory is intentionally distinct from allocator/device
        # evidence collected by DeviceMemoryTracker.  Baselines that do not
        # expose retained support memory record None per sample.
        'memory_bytes': expand('memory_bytes'),
        'admission_prediction': expand('admission_prediction'),
        'admission_normalized_entropy': expand('admission_normalized_entropy'),
        'admitted_to_memory': expand('admitted_to_memory'),
        'retrieval_profile': expand('retrieval_profile'),
        'retrieval_elapsed_ms': expand('retrieval_elapsed_ms'),
        'retrieval_candidate_count': expand('retrieval_candidate_count'),
        'retrieval_eligible_candidate_count': expand('retrieval_eligible_candidate_count'),
        'retrieval_returned_support_count': expand('retrieval_returned_support_count'),
        'retrieval_active_class_count': expand('retrieval_active_class_count'),
        'pre_adaptation_ood_score': expand('pre_adaptation_ood_score'),
        'retrieved_ood_fraction': expand('retrieved_ood_fraction'),
        'retrieved_ood_weight_fraction': expand('retrieved_ood_weight_fraction'),
        'ramen_vs_oracle_id_cosine': expand('ramen_vs_oracle_id_cosine'),
        'ramen_vs_oracle_id_sign_disagreement': expand('ramen_vs_oracle_id_sign_disagreement'),
        'consensus_mean_agreement': expand('consensus_mean_agreement'),
        'consensus_p10_agreement': expand('consensus_p10_agreement'),
        'consensus_p50_agreement': expand('consensus_p50_agreement'),
        'consensus_mask_rate': expand('consensus_mask_rate'),
        'consensus_active_class_count': expand('consensus_active_class_count'),
        'consensus_applied': expand('consensus_applied'),
    }


def _provide_oracle_domain_context(tta_model, domain_idx):
    """Hand evaluator domain IDs only to an explicitly opt-in diagnostic."""
    if not getattr(tta_model, 'requires_oracle_domain_context', False):
        return
    set_context = getattr(tta_model, 'set_oracle_domain_context', None)
    if not callable(set_context):
        raise RuntimeError('oracle-context method has no context hook')
    set_context(domain_idx)


def _provide_oracle_ood_context(tta_model, is_ood):
    """Hand evaluator ID/OOD flags only to an explicitly named oracle diagnostic."""
    if not getattr(tta_model, 'requires_oracle_ood_context', False):
        return
    set_context = getattr(tta_model, 'set_oracle_is_ood', None)
    if not callable(set_context):
        raise RuntimeError('oracle-OOD method has no context hook')
    set_context(is_ood)


def _evidence_paths(args):
    if not RUN_ID_PATTERN.fullmatch(args.run_id) or '..' in args.run_id:
        raise ValueError(
            'run_id must be 1-128 path-safe characters and may contain only '
            'letters, digits, dot, underscore, and hyphen'
        )
    run_dir = Path(args.evidence_dir) / args.run_id
    return {
        'run_dir': run_dir,
        'manifest': run_dir / 'manifest.json',
        'trace': run_dir / 'trace.jsonl',
        'summary': run_dir / 'summary.json',
        'stream': run_dir / 'stream.json',
    }


def _load_noadapt_reference_config(args):
    """Load the canonical NoAdapt config independently of the adapted method."""
    config_dir = getattr(args, 'config_dir', None)
    if config_dir is None:
        current_config_path = getattr(args, 'config_path', None)
        config_dir = Path(current_config_path).resolve().parents[1] if current_config_path else PROJECT_ROOT / 'cfg'
    config_dir = Path(config_dir).expanduser().resolve()
    candidates = (
        config_dir / args.dataset / 'NoAdapt.yaml',
        config_dir / 'default' / 'NoAdapt.yaml',
    )
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            with path.open('r', encoding='utf-8') as config_file:
                config = yaml.safe_load(config_file) or {}
            if not isinstance(config, dict):
                raise ValueError(f'NoAdapt config must contain a mapping: {path}')
            return config, str(path.resolve())
    raise ValueError(f'NoAdapt reference config is missing below {config_dir}')


def _reference_run_identity(args, artifacts):
    if artifacts.get('status') != 'verified' or artifacts.get('mode') not in {'fast', 'exact'}:
        raise ProvenanceError(
            'negative-adaptation references require --artifact-provenance fast or exact'
        )
    reference_config, reference_config_path = _load_noadapt_reference_config(args)
    return {
        'dataset': args.dataset,
        'model': args.model,
        'device': str(args.device),
        'data_root': str(Path(args.data_root).expanduser().resolve()),
        'tta_mode': args.tta_mode,
        'batch_size': args.batch_size,
        'metric_window_size': args.metric_window_size,
        'metric_window_stride': args.metric_window_stride,
        'stream_block_size': args.stream_block_size,
        'artifact_provenance': args.artifact_provenance,
        'artifacts': artifacts,
        'reference_config': reference_config,
        'reference_config_path': reference_config_path,
    }


def ordered_stream_test(
    datasets, tta_model, args, evidence_paths, stream_dataset, segments=None,
    reference_identity=None,
):
    reference_sha256 = None
    if args.reference_trace is not None:
        reference_sha256 = verify_reference_trace_stream_fingerprint(
            args.reference_trace,
            stream_dataset.fingerprint,
            expected_identity=reference_identity,
        )
    with evidence_paths['stream'].open('w', encoding='utf-8') as stream_file:
        json.dump(stream_dataset.to_dict(), stream_file, indent=2, sort_keys=True)
        stream_file.write('\n')

    dataset_num_corrects = torch.zeros(len(datasets), dtype=torch.int)
    dataset_num_samples = np.bincount(
        [domain_idx for domain_idx, _ in stream_dataset.references], minlength=len(datasets)
    )
    open_set_stream = stream_dataset.metadata.get('open_set')
    if open_set_stream is not None and not isinstance(open_set_stream, dict):
        raise ValueError('open-set stream metadata must be a mapping')
    id_domain_corrects = torch.zeros(len(datasets), dtype=torch.int) if open_set_stream else None
    id_domain_samples = torch.zeros(len(datasets), dtype=torch.int) if open_set_stream else None
    open_set_rows = []

    if segments is None:
        evaluation_parts = [stream_dataset]
    else:
        evaluation_parts = [Subset(stream_dataset, range(start, stop)) for start, stop in segments]

    trace_writer = JsonlTraceWriter(evidence_paths['trace'], args.run_id)
    timestep = 0
    total_correct = 0
    correctness_history = bytearray()
    domain_history = array('I')
    inferred_context_history = []
    window_values = deque(maxlen=args.metric_window_size)
    sliding_windows = []
    forward_latencies_ms = []
    retained_memory_bytes = []
    memory_bytes_available = None
    admission_rows = []
    admission_fields_available = None
    retrieval_profile_rows = []
    retrieval_profile_available = None
    oracle_gradient_rows = []
    consensus_rows = []
    consensus_diagnostics_available = None
    memory_tracker = DeviceMemoryTracker(args.device)
    memory_tracker.start()

    try:
        for part_index, evaluation_part in enumerate(evaluation_parts):
            dataloader = DataLoader(
                evaluation_part,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                # Even shuffle=False DataLoaders draw a worker base seed when
                # their iterator is created.  The legacy replay specifically
                # keeps that bookkeeping draw off global RNG; normal stream
                # modes retain their existing evaluator behavior.
                generator=(
                    torch.Generator().manual_seed(0)
                    if getattr(args, 'legacy_mixed_order', False) else None
                ),
            )
            for batch in tqdm(dataloader):
                if open_set_stream is None:
                    image, label, domain_idx, sample_idx = batch
                    evaluator_metadata = None
                    original_label = label
                    is_ood = torch.zeros_like(label, dtype=torch.bool)
                else:
                    image, source_label, domain_idx, sample_idx, evaluator_metadata = batch
                    required_metadata = {
                        'original_label', 'known_label_or_minus_one', 'is_ood',
                    }
                    if not isinstance(evaluator_metadata, dict) or set(evaluator_metadata) != required_metadata:
                        raise ValueError('open-set batches must carry complete evaluator-only metadata')
                    original_label = evaluator_metadata['original_label']
                    label = evaluator_metadata['known_label_or_minus_one']
                    is_ood = evaluator_metadata['is_ood']
                    if not all(torch.is_tensor(value) for value in (original_label, label, is_ood)):
                        raise ValueError('collated open-set evaluator metadata must be tensors')
                    if not torch.equal(source_label.to(torch.long), original_label.to(torch.long)):
                        raise ValueError('open-set source label disagrees with evaluator metadata')
                    if bool(torch.any(is_ood.to(torch.bool) != (label.to(torch.long) == -1))):
                        raise ValueError('open-set ID/OOD flags disagree with known labels')
                image, label = image.to(args.device), label.to(args.device)

                # The evaluator's domain label is deliberately unavailable to
                # ordinary methods.  The explicitly named oracle diagnostic
                # receives it through its one-shot context hook only.
                _provide_oracle_domain_context(tta_model, domain_idx)
                _provide_oracle_ood_context(tta_model, is_ood)

                _sync_device(args.device)
                started = time.perf_counter()
                logits = tta_model(image)
                _sync_device(args.device)
                batch_latency_ms = (time.perf_counter() - started) * 1000.0

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    entropy = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)
                    correct = pred.eq(label)
                    is_correct = correct.cpu().int()
                    dataset_num_corrects.index_add_(0, domain_idx, is_correct)
                    if open_set_stream is not None:
                        id_mask = ~is_ood.to(torch.bool)
                        id_domain_corrects.index_add_(0, domain_idx[id_mask], is_correct[id_mask])
                        id_domain_samples.index_add_(0, domain_idx[id_mask], torch.ones_like(is_correct[id_mask]))

                if args.device.type == 'mps':
                    _sync_device(args.device)
                memory_tracker.sample_post_batch()

                batch_size = label.size(0)
                diagnostics = _method_diagnostics(tta_model, batch_size)
                batch_memory_bytes = diagnostics['memory_bytes']
                batch_availability = {value is not None for value in batch_memory_bytes}
                if len(batch_availability) != 1:
                    raise ValueError(
                        'diagnostic memory_bytes must be available for every sample or unavailable for every sample'
                    )
                batch_memory_available = batch_availability.pop()
                if memory_bytes_available is None:
                    memory_bytes_available = batch_memory_available
                elif memory_bytes_available != batch_memory_available:
                    raise ValueError(
                        'diagnostic memory_bytes availability changed within one run'
                    )
                latency_ms = batch_latency_ms / batch_size
                labels = label.detach().cpu().tolist()
                predictions = pred.detach().cpu().tolist()
                correctness = correct.detach().cpu().tolist()
                entropies = entropy.detach().cpu().tolist()
                domains = domain_idx.tolist()
                samples = sample_idx.tolist()
                originals = original_label.detach().cpu().tolist()
                known_labels = label.detach().cpu().tolist()
                ood_flags = is_ood.detach().cpu().tolist()
                admission_predictions = diagnostics['admission_prediction']
                admission_entropies = diagnostics['admission_normalized_entropy']
                admissions = diagnostics['admitted_to_memory']
                admission_available = {
                    value is not None
                    for values in (admission_predictions, admission_entropies, admissions)
                    for value in values
                }
                if admission_available not in ({False}, {True}):
                    raise ValueError('admission diagnostics must be available for every sample or unavailable for every sample')
                batch_admission_available = admission_available == {True}
                if admission_fields_available is None:
                    admission_fields_available = batch_admission_available
                elif admission_fields_available != batch_admission_available:
                    raise ValueError('admission diagnostics availability changed within one run')
                profile_fields = (
                    'retrieval_profile', 'retrieval_elapsed_ms', 'retrieval_candidate_count',
                    'retrieval_eligible_candidate_count', 'retrieval_returned_support_count',
                    'retrieval_active_class_count',
                )
                profile_available = {value is not None for field in profile_fields for value in diagnostics[field]}
                if profile_available not in ({False}, {True}):
                    raise ValueError('retrieval profile diagnostics must be available for every sample or unavailable for every sample')
                batch_profile_available = profile_available == {True}
                if retrieval_profile_available is None:
                    retrieval_profile_available = batch_profile_available
                elif retrieval_profile_available != batch_profile_available:
                    raise ValueError('retrieval profile diagnostics availability changed within one run')
                ood_scores = diagnostics['pre_adaptation_ood_score']
                if open_set_stream is not None and any(value is None for value in ood_scores):
                    raise ValueError(
                        'open-set evaluation requires a finite pre_adaptation_ood_score diagnostic from the method'
                    )
                oracle_gradient_available = bool(
                    getattr(tta_model, 'emits_oracle_gradient_diagnostics', False)
                )
                oracle_fields = (
                    'retrieved_ood_fraction', 'retrieved_ood_weight_fraction',
                    'ramen_vs_oracle_id_cosine', 'ramen_vs_oracle_id_sign_disagreement',
                )
                if oracle_gradient_available and any(
                    diagnostics[field][offset] is None
                    for field in oracle_fields[:2]
                    for offset in range(batch_size)
                ):
                    raise ValueError('oracle-gradient methods must emit retrieved OOD diagnostics')
                consensus_fields = (
                    'consensus_mean_agreement', 'consensus_p10_agreement',
                    'consensus_p50_agreement', 'consensus_mask_rate',
                    'consensus_active_class_count', 'consensus_applied',
                )
                consensus_available = {
                    value is not None
                    for field in consensus_fields
                    for value in diagnostics[field]
                }
                if consensus_available not in ({False}, {True}):
                    raise ValueError(
                        'consensus diagnostics must be available for every sample or unavailable for every sample'
                    )
                batch_consensus_available = consensus_available == {True}
                if consensus_diagnostics_available is None:
                    consensus_diagnostics_available = batch_consensus_available
                elif consensus_diagnostics_available != batch_consensus_available:
                    raise ValueError('consensus diagnostics availability changed within one run')

                for offset in range(batch_size):
                    row = {
                        'timestep': timestep,
                        'sample_idx': int(samples[offset]),
                        'ground_truth_domain': int(domains[offset]),
                        'ground_truth_class': int(
                            originals[offset] if open_set_stream is not None else labels[offset]
                        ),
                        'prediction': int(predictions[offset]),
                        'correct': bool(correctness[offset]),
                        'predicted_entropy': float(entropies[offset]),
                        'inferred_context': diagnostics['inferred_context'][offset],
                        'memory_size': int(diagnostics['memory_size'][offset]),
                        'num_active_contexts': diagnostics['num_active_contexts'][offset],
                        'memory_bytes': diagnostics['memory_bytes'][offset],
                        'latency_ms': latency_ms,
                    }
                    if admission_available == {True}:
                        row.update({
                            'admission_prediction': admission_predictions[offset],
                            'admission_normalized_entropy': admission_entropies[offset],
                            'admitted_to_memory': admissions[offset],
                        })
                        admission_rows.append(row)
                    if batch_profile_available:
                        row.update({field: diagnostics[field][offset] for field in profile_fields})
                        retrieval_profile_rows.append(row)
                    if open_set_stream is not None:
                        row.update({
                            'original_label': int(originals[offset]),
                            'known_label_or_minus_one': int(known_labels[offset]),
                            'is_ood': bool(ood_flags[offset]),
                            'open_set_split_version': open_set_stream['split_version'],
                            'ood_ratio': float(open_set_stream['requested_ood_ratio']),
                            'pre_adaptation_ood_score': float(ood_scores[offset]),
                        })
                        open_set_rows.append(row)
                    if oracle_gradient_available:
                        row.update({field: diagnostics[field][offset] for field in oracle_fields})
                        oracle_gradient_rows.append(row)
                    if batch_consensus_available:
                        row.update({field: diagnostics[field][offset] for field in consensus_fields})
                        consensus_rows.append(row)
                    trace_writer.write(row)
                    forward_latencies_ms.append(latency_ms)
                    if row['memory_bytes'] is not None:
                        retained_memory_bytes.append(row['memory_bytes'])
                    total_correct += int(row['correct'])
                    correctness_history.append(int(row['correct']))
                    domain_history.append(row['ground_truth_domain'])
                    inferred_context_history.append(row['inferred_context'])
                    window_values.append(row['correct'])
                    if (len(window_values) == args.metric_window_size and
                            (timestep - args.metric_window_size + 1) % args.metric_window_stride == 0):
                        sliding_windows.append({
                            'start_timestep': timestep - args.metric_window_size + 1,
                            'end_timestep': timestep,
                            'accuracy': sum(window_values) / len(window_values),
                        })
                    timestep += 1
            if segments is not None and part_index < len(evaluation_parts) - 1:
                tta_model.reset()
    finally:
        trace_writer.close()

    dataset_num_corrects = dataset_num_corrects.numpy()
    dataset_accs = np.divide(
        dataset_num_corrects,
        dataset_num_samples,
        out=np.full(len(datasets), np.nan, dtype=float),
        where=dataset_num_samples > 0,
    )

    domain_names = datasets.environments

    device_memory = memory_tracker.summary()
    peak_memory_bytes = (
        device_memory['bytes'] if device_memory['kind'] == 'exact_cuda_allocator_peak' else None
    )
    if memory_bytes_available:
        method_memory = {
            'status': 'computed',
            'definition': 'exact bytes retained by the method support memory after each completed method forward; a batch snapshot is repeated for its samples',
            'unit': 'bytes',
            'max_retained_bytes': max(retained_memory_bytes),
            'final_retained_bytes': retained_memory_bytes[-1],
        }
    else:
        method_memory = {
            'status': 'unavailable',
            'reason': 'method did not expose retained support-memory bytes',
            'unit': 'bytes',
            'max_retained_bytes': None,
            'final_retained_bytes': None,
        }
    if forward_latencies_ms:
        total_forward_latency_ms = sum(forward_latencies_ms)
        forward_latency = {
            'status': 'computed',
            'definition': 'per-sample share of synchronized tta_model forward wall-clock latency; includes adaptation and prediction',
            'unit': 'milliseconds',
            'total_ms': total_forward_latency_ms,
            'mean_per_sample_ms': total_forward_latency_ms / len(forward_latencies_ms),
            'median_per_sample_ms': statistics.median(forward_latencies_ms),
        }
        throughput = {
            'status': 'computed',
            'definition': 'completed samples divided by total synchronized tta_model forward wall-clock latency',
            'unit': 'samples_per_second',
            'samples_per_second': (
                len(forward_latencies_ms) * 1000.0 / total_forward_latency_ms
                if total_forward_latency_ms > 0 else None
            ),
        }
    else:
        forward_latency = {
            'status': 'not_applicable',
            'reason': 'stream contained no completed samples',
            'unit': 'milliseconds',
            'total_ms': None,
            'mean_per_sample_ms': None,
            'median_per_sample_ms': None,
        }
        throughput = {
            'status': 'not_applicable',
            'reason': 'stream contained no completed samples',
            'unit': 'samples_per_second',
            'samples_per_second': None,
        }

    valid_accs = dataset_accs[~np.isnan(dataset_accs)]
    open_set_summary = None
    if open_set_stream is not None:
        id_counts = id_domain_samples.numpy()
        id_corrects = id_domain_corrects.numpy()
        id_accuracies = np.divide(
            id_corrects,
            id_counts,
            out=np.full(len(datasets), np.nan, dtype=float),
            where=id_counts > 0,
        )
        metric_kwargs = {
            'predictions': [row['prediction'] for row in open_set_rows],
            'ground_truth_classes': [row['known_label_or_minus_one'] for row in open_set_rows],
            'is_ood': [row['is_ood'] for row in open_set_rows],
            'ood_scores': [row['pre_adaptation_ood_score'] for row in open_set_rows],
        }
        try:
            metrics = open_set_metrics(**metric_kwargs)
            detection = {
                'status': 'computed',
                'score': 'negative_logsumexp_pre_adaptation_logits',
                'id_accuracy': metrics.id_accuracy,
                'auroc': metrics.auroc,
                'fpr95': metrics.fpr_at_95_tpr,
                'fpr95_threshold': metrics.fpr95_threshold,
                'ood_recall_at_fpr95': metrics.ood_recall_at_fpr95,
                'h_score': metrics.h_score,
                'id_count': metrics.id_count,
                'ood_count': metrics.ood_count,
            }
        except ValueError as exc:
            id_flags = metric_kwargs['is_ood']
            detection = {
                'status': 'unavailable',
                'reason': str(exc),
                'score': 'negative_logsumexp_pre_adaptation_logits',
                'id_accuracy': id_accuracy(
                    metric_kwargs['predictions'], metric_kwargs['ground_truth_classes'], id_flags,
                ) if any(not flag for flag in id_flags) else None,
                'auroc': None,
                'fpr95': None,
                'fpr95_threshold': None,
                'ood_recall_at_fpr95': None,
                'h_score': None,
                'id_count': sum(not flag for flag in id_flags),
                'ood_count': sum(id_flags),
            }
        valid_id_accs = id_accuracies[~np.isnan(id_accuracies)]
        open_set_summary = {
            **detection,
            'split_version': open_set_stream['split_version'],
            'requested_ood_ratio': open_set_stream['requested_ood_ratio'],
            'realized_ood_ratio': open_set_stream['realized_ood_ratio'],
            'realized_ood_count': open_set_stream['realized_ood_count'],
            'realized_known_count': open_set_stream['realized_known_count'],
            'id_domain_accuracies': {
                name: (float(accuracy) if not np.isnan(accuracy) else None)
                for name, accuracy in zip(domain_names, id_accuracies)
            },
            'worst_domain_id_accuracy': (
                float(np.min(valid_id_accs)) if len(valid_id_accs) else None
            ),
        }
    if segments is not None:
        recovery = {
            'status': 'not_applicable',
            'reason': 'single-domain evaluation resets adaptation state at every domain boundary',
        }
    elif args.stream_mode in {'block', 'recurring', 'bursty'}:
        recovery = {
            'status': 'computed',
            'definition': 'full-window recovery within each persistent-domain episode',
            'window_size': args.metric_window_size,
            'shifts': domain_shift_recovery_times(
                [bool(value) for value in correctness_history],
                domain_history,
                window_size=args.metric_window_size,
            ),
        }
    else:
        recovery = {
            'status': 'not_applicable',
            'reason': 'stream does not define discrete persistent-domain episodes',
        }
    if args.reference_trace is not None:
        negative_adaptation = compare_trace_negative_adaptation(
            evidence_paths['trace'],
            args.reference_trace,
            window_size=args.metric_window_size,
            stride=args.metric_window_stride,
            _expected_reference_sha256=reference_sha256,
        )
    else:
        negative_adaptation = {
            'status': 'reference_required',
            'reason': 'pass --reference_trace from NoAdapt on the identical stream',
        }
    profile_rows = retrieval_profile_rows
    summary = {
        'schema_version': SUMMARY_SCHEMA_VERSION,
        'run_id': args.run_id,
        'num_samples': int(timestep),
        'micro_accuracy': total_correct / timestep if timestep else None,
        'macro_domain_accuracy': float(np.mean(valid_accs)) if len(valid_accs) else None,
        'worst_domain_accuracy': float(np.min(valid_accs)) if len(valid_accs) else None,
        'domain_accuracies': {
            name: (float(accuracy) if not np.isnan(accuracy) else None)
            for name, accuracy in zip(domain_names, dataset_accs)
        },
        'domain_sample_counts': {
            name: int(count) for name, count in zip(domain_names, dataset_num_samples)
        },
        'sliding_window': {
            'window_size': args.metric_window_size,
            'stride': args.metric_window_stride,
            'values': sliding_windows,
        },
        'post_shift_recovery_time': recovery,
        'negative_adaptation_rate': negative_adaptation,
        'routing_diagnostics': asdict(
            routing_diagnostics(domain_history, inferred_context_history)
        ),
        'peak_device_memory_bytes': peak_memory_bytes,
        'device_memory': device_memory,
        'method_memory': method_memory,
        'forward_latency': forward_latency,
        'throughput': throughput,
        'retrieval_latency': {
            'status': 'unavailable',
            'reason': 'retrieval is interleaved with causal insertion and adaptation; isolating it would require invasive instrumentation and device synchronization that would perturb the measured path',
        },
        'stream_fingerprint': stream_dataset.fingerprint,
    }
    if open_set_summary is not None:
        summary['open_set'] = open_set_summary
    if oracle_gradient_rows:
        def oracle_mean(field, transform=lambda value: value):
            values = [transform(row[field]) for row in oracle_gradient_rows if row[field] is not None]
            return sum(values) / len(values) if values else None
        summary['oracle_gradient_diagnostics'] = {
            'status': 'computed',
            'retrieved_ood_fraction_mean': oracle_mean('retrieved_ood_fraction'),
            'retrieved_ood_weight_fraction_mean': oracle_mean('retrieved_ood_weight_fraction'),
            'gradient_direction_corruption_mean': oracle_mean(
                'ramen_vs_oracle_id_cosine', lambda value: 1.0 - value,
            ),
            'sign_disagreement_mean': oracle_mean('ramen_vs_oracle_id_sign_disagreement'),
            'defined_direction_count': sum(
                row['ramen_vs_oracle_id_cosine'] is not None for row in oracle_gradient_rows
            ),
        }
    if consensus_rows:
        def consensus_mean(field):
            return sum(row[field] for row in consensus_rows) / len(consensus_rows)
        applied_consensus_rows = [row for row in consensus_rows if row['consensus_applied']]
        def applied_consensus_mean(field):
            if not applied_consensus_rows:
                return None
            return sum(row[field] for row in applied_consensus_rows) / len(applied_consensus_rows)
        summary['consensus_diagnostics'] = {
            'status': 'computed',
            'consensus_applied_sample_count': len(applied_consensus_rows),
            'consensus_applied_sample_fraction': len(applied_consensus_rows) / len(consensus_rows),
            # Agreement and retention describe actual hard-mask decisions,
            # not ordinary-Ramen fallback samples.
            'mean_agreement': applied_consensus_mean('consensus_mean_agreement'),
            'p10_agreement': applied_consensus_mean('consensus_p10_agreement'),
            'p50_agreement': applied_consensus_mean('consensus_p50_agreement'),
            'mask_rate': applied_consensus_mean('consensus_mask_rate'),
            'mean_active_class_count': applied_consensus_mean('consensus_active_class_count'),
            'all_sample_mean_active_class_count': consensus_mean('consensus_active_class_count'),
        }
    if profile_rows:
        values = sorted(row['retrieval_elapsed_ms'] for row in profile_rows)
        def percentile(fraction):
            position = (len(values) - 1) * fraction
            lower, upper = int(position), min(int(position) + 1, len(values) - 1)
            return values[lower] + (values[upper] - values[lower]) * (position - lower)
        def distribution(field):
            data = [row[field] for row in profile_rows]
            return {'min': min(data), 'p50': percentile_from(data, .5), 'p95': percentile_from(data, .95), 'max': max(data)}
        def percentile_from(data, fraction):
            ordered = sorted(data); position = (len(ordered) - 1) * fraction
            lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        summary['retrieval_latency'] = {
            'status': 'computed',
            'profile': 'causal_sync_v1',
            'definition': 'device-synchronized query-only interval after causal insertion; synchronization perturbs execution and is not comparable to ordinary end-to-end latency',
            'unit': 'milliseconds', 'total_ms': sum(values), 'p50_ms': percentile(.5), 'p95_ms': percentile(.95), 'max_ms': max(values),
            'candidate_count': distribution('retrieval_candidate_count'),
            'eligible_candidate_count': distribution('retrieval_eligible_candidate_count'),
            'returned_support_count': distribution('retrieval_returned_support_count'),
            'active_class_count': distribution('retrieval_active_class_count'),
        }
    if admission_rows:
        admitted_rows = [row for row in admission_rows if row['admitted_to_memory']]
        rejected_rows = [row for row in admission_rows if not row['admitted_to_memory']]
        def pseudo_accuracy(rows):
            return (sum(row['admission_prediction'] == row['ground_truth_class'] for row in rows) / len(rows)) if rows else None
        summary['admission_diagnostics'] = {
            'admitted_count': len(admitted_rows),
            'rejected_count': len(rejected_rows),
            'admission_rate': len(admitted_rows) / len(admission_rows),
            'mean_normalized_entropy': sum(row['admission_normalized_entropy'] for row in admission_rows) / len(admission_rows),
            'admitted_pseudo_label_accuracy': pseudo_accuracy(admitted_rows),
            'rejected_pseudo_label_accuracy': pseudo_accuracy(rejected_rows),
            'admitted_contamination_rate': (
                1.0 - pseudo_accuracy(admitted_rows) if admitted_rows else None
            ),
        }
    write_summary(evidence_paths['summary'], summary)

    tta_model.reset()

    return domain_names, dataset_accs, stream_dataset.metadata


def main(args):
    evidence_paths = _evidence_paths(args)
    open_set = bool(getattr(args, 'open_set', False))
    print('-' * 80)
    print(args)
    print('-' * 80)
    print(args.config)
    print('-' * 80)

    artifacts = _artifact_provenance(args)
    print("Loading model...")
    verified_checkpoint_path = (
        artifacts['model']['path'] if artifacts['status'] == 'verified' else None
    )
    model, preprocess = get_pretrained_model(
        args, verified_checkpoint_path=verified_checkpoint_path
    )

    print("Loading datasets...")
    if open_set:
        dataset_class = _open_set_dataset_class(args.dataset, args.known_class_split)
        datasets = dataset_class(
            root=args.data_root, transform=preprocess,
            split_path=getattr(args, 'known_class_split_path', None),
        )
    else:
        datasets = get_dataset_class(args.dataset)(root=args.data_root, transform=preprocess)
    print(f"dataset includes environments: \n{datasets.environments}")
    artifacts = _revalidate_artifact_provenance(args, artifacts)
    reference_identity = (
        _reference_run_identity(args, artifacts) if args.reference_trace is not None else None
    )

    method_class = get_method_class(args.tta_algo)
    tta_method = None

    stream_dataset = None
    segments = None
    if args.tta_mode == 'mixed':
        if args.legacy_mixed_order:
            # Exact historical sequencing: method construction happened before
            # the shuffled DataLoader iterator consumed its two global seeds.
            print("Initializing model before historical legacy-order replay")
            tta_method = method_class(model, datasets, args)
            stream_dataset = build_legacy_torch_iid_stream(datasets, args.seed)
        else:
            build_kwargs = {
                'mode': args.stream_mode,
                'seed': args.stream_seed,
                'domain_weights': args.stream_domain_weights,
                'block_size': args.stream_block_size,
                'gradual_sharpness': args.stream_gradual_sharpness,
                'sample_budget': args.stream_sample_budget,
                'novel_domain_idx': args.stream_novel_domain_idx,
                'novel_release_fraction': args.stream_novel_release_fraction,
                'correlation_strength': args.stream_correlation_strength,
                'burst_size': args.stream_burst_size,
            }
            stream_dataset = (
                build_open_set_stream(
                    datasets, ood_ratio=getattr(args, 'ood_ratio', 0.0),
                    per_domain_source_budget=args.open_set_per_domain_source_budget,
                    **build_kwargs,
                )
                if open_set else build_stream(datasets, **build_kwargs)
            )
    elif args.tta_mode == 'single':
        stream_dataset, segments = build_single_domain_stream(datasets, args.stream_seed)
    else:
        raise ValueError(f'Unknown tta mode: {args.tta_mode}')

    if args.max_eval_samples is not None:
        stream_dataset = truncate_stream(stream_dataset, args.max_eval_samples)
        if segments is not None:
            segments = _prefix_segments(segments, len(stream_dataset))

    evidence_paths['run_dir'].mkdir(parents=True, exist_ok=False)

    manifest_args = {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items()
        if key != 'config'
    }
    manifest_args['oracle_domain_contexts'] = bool(
        getattr(method_class, 'requires_oracle_domain_context', False)
    )
    manifest_args['oracle_ood_contexts'] = bool(
        getattr(method_class, 'requires_oracle_ood_context', False)
    )
    manifest_args['data_root'] = str(Path(args.data_root).expanduser().resolve())
    if manifest_args.get('config_path') is not None:
        manifest_args['config_path'] = str(Path(manifest_args['config_path']).expanduser().resolve())
    write_run_manifest(
        evidence_paths['manifest'],
        run_id=args.run_id,
        args=manifest_args,
        config=args.config,
        device=args.device,
        dataset={
            'name': args.dataset,
            'environments': list(datasets.environments),
            'original_domain_lengths': [len(dataset) for dataset in datasets],
            'open_set': stream_dataset.metadata.get('open_set'),
        },
        stream=stream_dataset.metadata,
        artifacts=artifacts,
        hardware=collect_hardware_evidence(
            'cuda' if args.cuda else args.device_request, args.device
        ),
        repository=PROJECT_ROOT,
        package_names=['torch', 'torchvision', 'numpy', 'PyYAML', 'Pillow', 'tqdm', 'clip'],
    )

    if tta_method is None:
        print("Initializing model")
        tta_method = method_class(model, datasets, args)

    if args.tta_mode in {'single', 'mixed'}:
        domain_names, dataset_accs, _ = ordered_stream_test(
            datasets, tta_method, args, evidence_paths, stream_dataset, segments,
            reference_identity=reference_identity,
        )

    else:
        raise ValueError(f'Unknown tta mode: {args.tta_mode}')

    # Print results
    print('-' * 80)
    for env, acc in zip(domain_names, dataset_accs):
        print(f"{env}: {acc * 100:.2f}%")

    print('-' * 80)
    print(f"total: {np.nanmean(dataset_accs) * 100:.2f}%")

    # SAVE csv here if needed
    csv_path = args.save_to
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)

        args_dict = {k: str(v) for k, v in vars(args).items() if isinstance(v, (int, str, float, bool))}
        writer.writerow([f"{k}={v}" for k, v in args_dict.items()])

        writer.writerow([f"{k}={v}" for k, v in args.config.items()])

        writer.writerow(datasets.environments + ["Average"])

        avg_acc = np.nanmean(dataset_accs)
        accs_formatted = [f"{acc * 100:.2f}" if not np.isnan(acc) else "" for acc in dataset_accs]
        writer.writerow(accs_formatted + [f"{avg_acc * 100:.2f}"])

        writer.writerow([])


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CIFAR10C')

    parser.add_argument('--data_root', type=str, default='~/data')

    parser.add_argument('--artifact-provenance', choices=['off', 'fast', 'exact'], default='off',
                        help='verify cached official CLIP and canonical dataset provenance before a run')

    parser.add_argument('--tta_mode', type=str, default='mixed', choices=['single', 'mixed'])
    parser.add_argument(
        '--open_set', action='store_true', default=False,
        help='use a versioned CIFAR-100-C or DomainNet semantic open-set evaluator protocol',
    )
    parser.add_argument(
        '--known_class_split', type=str, default='open-set-cifar100-split-v1',
        help='versioned open-set split identifier for the selected dataset',
    )
    parser.add_argument(
        '--ood_ratio', type=float, default=0.0,
        help='exact per-domain OOD fraction for an open-set stream',
    )

    parser.add_argument('--model', type=str, default='clip_vitbase32',
                        help='model name')

    parser.add_argument('--use_tbn', action='store_true', default=False,
                        help='whether to use target batch norm statistics')

    parser.add_argument('--tta_algo', type=str, default='NoAdapt',
                        help='tta algorithm name')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of images in each mini-batch')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for dataloader')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='deprecated alias for --device cuda')

    parser.add_argument('--device', dest='device_request', choices=['auto', 'cpu', 'cuda', 'mps'],
                        default='auto', help='execution backend; auto prefers CUDA, then MPS, then CPU')

    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads')

    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'cfg'))

    parser.add_argument('--save_to', type=str, default=str(PROJECT_ROOT / 'log' / 'default.csv'))

    parser.add_argument('--stream_mode', type=str, default='iid_mixed', choices=[
        'iid_mixed', 'block', 'gradual', 'recurring', 'imbalanced',
        'novel_domain', 'class_domain_correlated', 'bursty',
    ])
    parser.add_argument(
        '--legacy_mixed_order', action='store_true', default=False,
        help=(
            'exactly replay historical PyTorch DataLoader(generator=None) after '
            'method construction; constructor RNG may make order method-dependent '
            'and requires --num_workers 0 and --stream_seed equal to --seed'
        ),
    )
    parser.add_argument('--stream_seed', type=int, default=None)
    parser.add_argument('--stream_domain_weights', type=_parse_domain_weights, default=None)
    parser.add_argument('--stream_block_size', type=int, default=64)
    parser.add_argument('--stream_gradual_sharpness', type=float, default=4.0)
    parser.add_argument('--stream_sample_budget', type=int, default=None)
    parser.add_argument(
        '--open_set_per_domain_source_budget', '--open-set-per-domain-source-budget',
        dest='open_set_per_domain_source_budget', type=int, default=None,
        help='exact source examples to select from each domain before open-set scheduling',
    )
    parser.add_argument('--max_eval_samples', '--max-eval-samples', dest='max_eval_samples',
                        type=int, default=None,
                        help='deterministic stream-prefix budget for cost-limited evaluation')
    parser.add_argument('--stream_novel_domain_idx', type=int, default=None)
    parser.add_argument('--stream_novel_release_fraction', type=float, default=0.5)
    parser.add_argument('--stream_correlation_strength', type=float, default=0.9)
    parser.add_argument('--stream_burst_size', type=int, default=None)

    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--evidence_dir', type=str, default=str(PROJECT_ROOT / 'evidence'))
    parser.add_argument('--reference_trace', type=str, default=None,
                        help='NoAdapt trace on the identical stream for negative-adaptation rate')
    parser.add_argument('--metric_window_size', type=int, default=50)
    parser.add_argument('--metric_window_stride', type=int, default=50)

    args = parser.parse_args()

    args.data_root = str(Path(args.data_root).expanduser().resolve())
    if args.open_set:
        if args.tta_mode != 'mixed':
            parser.error('--open_set requires --tta_mode mixed')
        if args.legacy_mixed_order:
            parser.error('--open_set cannot use --legacy_mixed_order')
        try:
            _open_set_dataset_class(args.dataset, args.known_class_split)
        except ValueError as exc:
            parser.error(str(exc))
        if not 0.0 <= args.ood_ratio <= 1.0:
            parser.error('--ood_ratio must be between 0 and 1')
        if (args.open_set_per_domain_source_budget is not None
                and args.open_set_per_domain_source_budget <= 0):
            parser.error('--open_set_per_domain_source_budget must be a positive integer')
        args.known_class_split_path = str(
            PROJECT_ROOT / 'cfg' / 'research'
            / OPEN_SET_SPLIT_FILENAMES[(args.dataset, args.known_class_split)]
        )
    else:
        if args.ood_ratio != 0.0:
            parser.error('--ood_ratio requires --open_set')
        if args.open_set_per_domain_source_budget is not None:
            parser.error('--open_set_per_domain_source_budget requires --open_set')
        args.known_class_split_path = None
    if args.tta_algo in {'OracleIDGradientRamen', 'OracleDropOODRamen', 'OracleConsensusRamen'} and not args.open_set:
        parser.error(f'--tta_algo {args.tta_algo} requires --open_set evaluator labels')

    if args.cuda and args.device_request not in ('auto', 'cuda'):
        parser.error('--cuda cannot be combined with a non-CUDA --device')
    requested_device = 'cuda' if args.cuda else args.device_request
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if requested_device == 'auto':
        requested_device = 'cuda' if torch.cuda.is_available() else 'mps' if mps_available else 'cpu'
    if requested_device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but torch.cuda.is_available() is false')
    if requested_device == 'mps' and not mps_available:
        parser.error('MPS was requested but torch.backends.mps.is_available() is false')
    args.device = torch.device(requested_device)
    if args.stream_seed is None:
        args.stream_seed = args.seed
    if args.legacy_mixed_order and args.stream_seed != args.seed:
        parser.error(
            '--legacy_mixed_order uses historical global RNG initialized by '
            '--seed, so --stream_seed must equal --seed'
        )
    if args.legacy_mixed_order and args.num_workers != 0:
        parser.error('--legacy_mixed_order requires --num_workers 0')
    if args.legacy_mixed_order and args.reference_trace is not None:
        parser.error(
            '--legacy_mixed_order is historical parity and cannot be used for '
            'negative-adaptation pairing; use seeded --stream_mode iid_mixed'
        )
    if args.stream_block_size <= 0:
        parser.error('--stream_block_size must be a positive integer')
    if args.stream_block_size != 64 and args.max_eval_samples is None:
        parser.error('nondefault --stream_block_size requires --max_eval_samples')
    if args.run_id is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        stream_identity = (
            'legacy-torch-iid-replay' if args.legacy_mixed_order else args.stream_mode
        )
        if args.open_set:
            stream_identity = f'{stream_identity}-open-{args.known_class_split}-ood{args.ood_ratio:g}'
        block_identity = '' if args.stream_block_size == 64 else f'-blk-{args.stream_block_size}'
        args.run_id = f'{args.dataset}-{args.tta_algo}-{stream_identity}-s{args.seed}{block_identity}-{timestamp}'
    if args.metric_window_size <= 0 or args.metric_window_stride <= 0:
        parser.error('metric window size and stride must be positive')
    if args.legacy_mixed_order and (
        args.tta_mode != 'mixed' or args.stream_mode != 'iid_mixed'
    ):
        parser.error('--legacy_mixed_order requires --tta_mode mixed --stream_mode iid_mixed')

    args.config_dir = str(Path(args.config).expanduser().resolve())
    filepaths = [
        os.path.join(args.config_dir, args.dataset, args.tta_algo + '.yaml'),
        os.path.join(args.config_dir, 'default', args.tta_algo + '.yaml'),
    ]

    found_yaml = False

    args.config_path = None
    for filepath in filepaths:
        if os.path.exists(filepath):
            print('Loading config from {}'.format(filepath))
            with open(filepath, 'r') as f:
                args.config = yaml.safe_load(f)
            args.config_path = filepath

            found_yaml = True
            break

    if not found_yaml:
        args.config = {}

    args.max_batch_size = args.batch_size

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    torch.set_num_threads(args.num_threads)
    setup_seed(args.seed)
    main(args)
