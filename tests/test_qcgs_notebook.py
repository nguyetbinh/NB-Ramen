"""Validate actual notebook generation and checkpoint IO without claiming GPU evidence."""
import base64
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
