import io
import json
from pathlib import Path
import tempfile
import unittest

from data import DatasetWorkspace, DatasetManifest, load_manifest


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = DatasetWorkspace(self.root/'datasets', chunk_size=3)

    def upload(self, **kwargs):
        return self.ws.import_upload(io.BytesIO(b'x,y\n0,a\n1,b\n'), filename='data.csv',
                                     dataset_id='example', **kwargs)

    def test_constructor_is_lazy(self):
        self.assertEqual(self.ws.list_datasets(), [])
        self.assertFalse(self.ws.root.exists())

    def test_import_files_and_manifest_roundtrip(self):
        paths = [self.root/'first file.csv', self.root/'second.csv']
        for p in paths:
            p.write_bytes(b'a\r\n1\r\n')
        ds = self.ws.import_files(paths, name='My dataset', dataset_id='example')
        self.assertEqual([p.read_bytes() for p in ds.raw_files], [p.read_bytes() for p in paths])
        self.assertEqual(self.ws.get('example', verify=True).manifest, ds.manifest)
        self.assertIsInstance(load_manifest(ds.directory/'manifest.json'), DatasetManifest)
        self.assertTrue(ds.profiles_dir.is_dir())
        self.assertTrue(ds.recipes_dir.is_dir())
        self.assertEqual(self.ws.list_versions('example'), [])

    def test_upload_is_bounded_and_not_closed(self):
        class BoundedStream(io.BytesIO):
            def read(inner, size=-1):
                self.assertEqual(size, 3)
                return super().read(size)
        stream = BoundedStream(b'123456789')
        ds = self.ws.import_upload(stream, filename='upload.bin')
        self.assertEqual(ds.raw_files[0].read_bytes(), b'123456789')
        self.assertFalse(stream.closed)

    def test_stream_position_is_respected(self):
        stream = io.BytesIO(b'prefixDATA')
        stream.seek(6)
        ds = self.ws.import_upload(stream, filename='upload.bin')
        self.assertEqual(ds.raw_files[0].read_bytes(), b'DATA')

    def test_duplicate_import_does_not_overwrite(self):
        ds = self.upload()
        before = ds.raw_files[0].read_bytes()
        with self.assertRaises(FileExistsError):
            self.upload()
        self.assertEqual(ds.raw_files[0].read_bytes(), before)

    def test_limit_and_failed_stream_cleanup(self):
        with self.assertRaises(ValueError):
            self.upload(max_bytes=2)
        self.assertFalse((self.ws.root/'example').exists())
        self.assertFalse(list(self.ws.root.glob('.import-*')))
        with self.assertRaises(TypeError):
            self.ws.import_upload(io.StringIO('text'), filename='data.csv', dataset_id='example')
        self.upload()  # Failure released the lock.

    def test_multiple_file_limit_is_total(self):
        a, b = self.root/'a.csv', self.root/'b.csv'
        a.write_bytes(b'123'); b.write_bytes(b'456')
        with self.assertRaises(ValueError):
            self.ws.import_files([a, b], max_bytes=5, dataset_id='limited')
        self.assertFalse((self.ws.root/'limited').exists())

    def test_paths_and_duplicate_filenames_rejected(self):
        for name in ('../escape', 'CON', 'a/b', 'a:b', 'bad.'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.ws.import_upload(io.BytesIO(b'x'), filename=name)
        with self.assertRaises(ValueError):
            self.ws.import_upload(io.BytesIO(b'x'), filename='ok', dataset_id='../outside')
        a = self.root/'a.csv'; a.write_bytes(b'x')
        with self.assertRaises(ValueError):
            self.ws.import_files([a, a])

    def test_tampered_raw_detection(self):
        ds = self.upload()
        ds.raw_files[0].write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.ws.get(ds.dataset_id, verify=True)

    def test_manifest_rejects_unsupported_versions_and_traversal(self):
        ds = self.upload()
        path = ds.directory/'manifest.json'
        original = json.loads(path.read_text())
        for changes in ({'format_version': 99}, {'files': [{'path': '../escape', 'size_bytes': 0, 'sha256': '0'*64}]}):
            path.write_text(json.dumps({**original, **changes}))
            with self.assertRaises(ValueError):
                self.ws.get('example')

    def test_schema_update_preserves_raw(self):
        ds = self.upload()
        updated = self.ws.set_schema('example', {'x': 'int64', 'y': 'string'})
        self.assertEqual(updated.manifest.files, ds.manifest.files)
        self.assertEqual(self.ws.get('example').manifest.schema['x'], 'int64')
        with self.assertRaises(ValueError):
            self.ws.set_schema('example', {'x': 1})
        self.assertEqual(self.ws.get('example').manifest.schema['x'], 'int64')

    def test_processed_version_publication_and_lineage(self):
        ds = self.upload()
        with self.ws.processed_version('example', recipe={'steps': []}) as draft:
            self.assertEqual(draft.version, 'v1')
            self.assertEqual(self.ws.list_versions('example'), [])
            (draft.data_dir/'part-001.csv').write_bytes(b'x\n1\n')
            draft.schema['x'] = 'int64'
        self.assertEqual(self.ws.list_versions('example'), ['v1'])
        first = self.ws.get_version('example', 'v1', verify=True)
        self.assertEqual(first.schema, {'x': 'int64'})
        self.assertEqual(first.source_version, 'raw')
        with self.ws.processed_version('example', recipe={'steps': []}, source_version='v1') as draft:
            (draft.data_dir/'part.csv').write_bytes(b'x\n2\n')
        self.assertEqual(self.ws.get_version('example', 'v2').source_version, 'v1')
        self.assertEqual(self.ws.get('example', verify=True).manifest.files, ds.manifest.files)

    def test_failed_and_empty_versions_not_published(self):
        self.upload()
        with self.assertRaises(RuntimeError):
            with self.ws.processed_version('example', recipe={}) as draft:
                (draft.data_dir/'partial.csv').write_bytes(b'x')
                raise RuntimeError('executor failed')
        with self.assertRaises(ValueError):
            with self.ws.processed_version('example', recipe={}):
                pass
        self.assertEqual(self.ws.list_versions('example'), [])
        self.assertFalse(list((self.ws.root/'example'/'processed').glob('.version-*')))

    def test_processed_checksums_and_recipe_immutable(self):
        self.upload()
        with self.assertRaises(ValueError):
            with self.ws.processed_version('example', recipe={}) as draft:
                (draft.data_dir/'part.csv').write_bytes(b'x')
                (draft.directory/'recipe.json').write_text('{"changed":true}')
        with self.ws.processed_version('example', recipe={}) as draft:
            (draft.data_dir/'part.csv').write_bytes(b'x')
        (self.ws.root/'example'/'processed'/'v1'/'data'/'part.csv').write_bytes(b'y')
        with self.assertRaises(ValueError):
            self.ws.get_version('example', 'v1', verify=True)

    def test_concurrent_writer_rejected_and_lock_released(self):
        self.upload()
        with self.ws.processed_version('example', recipe={}) as draft:
            with self.assertRaises(RuntimeError):
                self.ws.set_schema('example', {})
            (draft.data_dir/'out.csv').write_bytes(b'x')
        self.ws.set_schema('example', {})


if __name__ == '__main__':
    unittest.main()
