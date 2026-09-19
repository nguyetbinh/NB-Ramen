"""Validate actual notebook generation and checkpoint IO without claiming GPU evidence."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded


class NotebookTests(unittest.TestCase):
    def test_git_setup_restores_exact_source_and_rejects_dirty_checkout(self):
        builder=module('qcgs_git_builder',ROOT/'notebooks/kaggle/build-qcgs-notebook.py')
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
        nb=builder.build(revision,rescue=True,repository_url=ROOT.as_uri(),source_tree=tree)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'checkout';evidence=root/'evidence'
            # Change only the output directories; execute the actual generated setup.
            setup=ast.parse(''.join(nb['cells'][1]['source']))
            for node in setup.body:
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
                    if node.targets[0].id in ('REPO','EVIDENCE'):
                        node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),
                            args=[ast.Constant(str(repo if node.targets[0].id=='REPO' else evidence))],keywords=[])
            code=compile(ast.fix_missing_locations(setup),'generated-git-setup','exec')
            exec(code,{})
            self.assertEqual(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),revision)
            self.assertEqual(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),'')
            receipt=json.loads((evidence/'runtime/source-git.json').read_text())
            self.assertEqual(receipt,dict(revision=revision,repository=ROOT.as_uri(),tree=tree))
            exec(code,{})  # idempotent setup keeps the exact existing checkout
            with (repo/'src/main.py').open('a') as handle:handle.write('\n# dirty checkout\n')
            with self.assertRaises(AssertionError):exec(code,{})
            # Failed verification must not publish a partial checkout.
            for node in setup.body:
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
                    if node.targets[0].id=='REPO':
                        node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(root/'bad'))],keywords=[])
                    elif node.targets[0].id=='SOURCE_TREE':node.value=ast.Constant('0'*40)
            with self.assertRaises(AssertionError):exec(compile(ast.fix_missing_locations(setup),'bad-tree-setup','exec'),{})
            self.assertFalse((root/'bad').exists())
            self.assertEqual(list(root.glob('qcgs-source-*')),[])

    def test_generated_size_limit_counts_utf8_and_preserves_previous_file(self):
        builder=module('qcgs_size_builder',ROOT/'notebooks/kaggle/build-qcgs-notebook.py')
        nb=builder.build('a'*40,rescue=True,repository_url='https://github.com/nguyetbinh/NB-Ramen',source_tree='b'*40)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'notebook.ipynb'
            builder.write_notebook(nb,output)
            before=output.read_bytes()
            self.assertLess(len(before),20_000)
            self.assertNotIn('SOURCE_BUNDLE',before.decode())
            self.assertIn('GO_CONFIRM',before.decode())
            nb['cells'][0]['source']=['é'*500_000]
            with self.assertRaisesRegex(ValueError,'Kaggle requires less than 1000000'):
                builder.write_notebook(nb,output)
            self.assertEqual(before,output.read_bytes())

    def test_embedded_source_bundle_round_trip_and_cell_syntax(self):
        builder=module('qcgs_builder',ROOT/'notebooks/kaggle/build-qcgs-notebook.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            subprocess.run(['git','init','-q',str(source)],check=True)
            (source/'real-file.txt').write_text('source fixture\n')
            subprocess.run(['git','add','real-file.txt'],cwd=source,check=True)
            subprocess.run(['git','-c','user.name=Test','-c','user.email=test@localhost','-c','commit.gpgsign=false','commit','-qm','test: source fixture'],cwd=source,check=True)
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
            bundle=root/'source.bundle'
            subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=source,check=True)
            nb=builder.build(revision,bundle.read_bytes())
            rescue=builder.build(revision,bundle.read_bytes(),rescue=True)
            self.assertEqual(rescue['metadata']['qcgs']['diagnostic'],'qcgs-multiview')
            for index,cell in enumerate(rescue['cells']):
                if cell['cell_type']=='code':
                    compile(''.join(cell['source']),f'multiview-cell-{index}','exec')
                    self.assertEqual(cell['outputs'],[])
            rescue_text='\n'.join(''.join(c['source']) for c in rescue['cells'])
            self.assertIn('GO_CONFIRM',rescue_text)
            self.assertIn('qcgs-multiview-evidence.zip',rescue_text)
            self.assertNotIn('qcgs-label-free-evidence',rescue_text)
            self.assertFalse(nb['metadata']['qcgs']['experimental_outputs'])
            for index,cell in enumerate(nb['cells']):
                if cell['cell_type']=='code':
                    compile(''.join(cell['source']),f'cell-{index}','exec')
                    self.assertEqual(cell['outputs'],[])
            target=root/'restored'
            subprocess.run(['git','clone','--quiet','--no-checkout',str(bundle),str(target)],check=True)
            subprocess.run(['git','checkout','--quiet','--detach',revision],cwd=target,check=True)
            self.assertEqual((target/'real-file.txt').read_text(),'source fixture\n')
            cells=[''.join(c['source']) for c in nb['cells']]
            self.assertIn('--preflight-only',cells[3]);self.assertIn('prepare-huggingface-data.py',cells[4])
            self.assertIn('--execute',cells[5]);self.assertIn('--audit',cells[6])

    def test_checkpoint_is_atomic_and_restore_preserves_progress(self):
        support=module('qcgs_checkpoint',ROOT/'notebooks/kaggle/full-run-checkpoint.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);evidence=root/'campaign';evidence.mkdir()
            (evidence/'row.json').write_text('{"complete":true}')
            archive=support.checkpoint(evidence)
            before=archive.read_bytes()
            (evidence/'bad-link').symlink_to(evidence/'row.json')
            with self.assertRaises(ValueError):support.checkpoint(evidence)
            self.assertEqual(archive.read_bytes(),before)
            destination=root/'resume'/'campaign'
            support.restore(archive,destination)
            (destination/'progress').write_text('retained')
            support.restore(archive,destination)
            self.assertTrue((destination/'progress').exists())


if __name__=='__main__':unittest.main()
