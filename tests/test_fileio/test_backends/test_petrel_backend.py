# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from shutil import SameFileError
from unittest import TestCase
from unittest.mock import MagicMock

from mmengine.fileio.backends import PetrelBackend


@contextmanager
def build_temporary_directory():
    """Build a temporary directory containing many files to test
    ``FileClient.list_dir_or_file``.

    . \n
    | -- dir1 \n
    | -- | -- text3.txt \n
    | -- dir2 \n
    | -- | -- dir3 \n
    | -- | -- | -- text4.txt \n
    | -- | -- img.jpg \n
    | -- text1.txt \n
    | -- text2.txt \n
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        text1 = Path(tmp_dir) / 'text1.txt'
        text1.open('w').write('text1')
        text2 = Path(tmp_dir) / 'text2.txt'
        text2.open('w').write('text2')
        dir1 = Path(tmp_dir) / 'dir1'
        dir1.mkdir()
        text3 = dir1 / 'text3.txt'
        text3.open('w').write('text3')
        dir2 = Path(tmp_dir) / 'dir2'
        dir2.mkdir()
        jpg1 = dir2 / 'img.jpg'
        jpg1.open('wb').write(b'img')
        dir3 = dir2 / 'dir3'
        dir3.mkdir()
        text4 = dir3 / 'text4.txt'
        text4.open('w').write('text4')
        yield tmp_dir


try:
    # Other unit tests may mock these modules so we need to pop them first.
    sys.modules.pop('petrel_client', None)
    sys.modules.pop('petrel_client.client', None)

    # If petrel_client is imported successfully, we can test PetrelBackend
    # without mock.
    import petrel_client  # noqa: F401
except ImportError:
    sys.modules['petrel_client'] = MagicMock()
    sys.modules['petrel_client.client'] = MagicMock()

else:

    class TestPetrelBackend(TestCase):  # type: ignore

        @classmethod
        def setUpClass(cls):
            cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
            cls.local_img_path = cls.test_data_dir / 'color.jpg'
            cls.local_img_shape = (300, 400, 3)
            cls.petrel_dir = 'petrel://mmengine-test/data'

        def setUp(self):
            backend = PetrelBackend()
            backend.rmtree(self.petrel_dir)
            with build_temporary_directory() as tmp_dir:
                backend.copytree_from_local(tmp_dir, self.petrel_dir)

            text1_path = f'{self.petrel_dir}/text1.txt'
            text2_path = f'{self.petrel_dir}/text2.txt'
            text3_path = f'{self.petrel_dir}/dir1/text3.txt'
            text4_path = f'{self.petrel_dir}/dir2/dir3/text4.txt'
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertTrue(backend.isfile(text1_path))
            self.assertTrue(backend.isfile(text2_path))
            self.assertTrue(backend.isfile(text3_path))
            self.assertTrue(backend.isfile(text4_path))
            self.assertTrue(backend.isfile(img_path))

        def test_get(self):
            backend = PetrelBackend()
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertEqual(backend.get(img_path), b'img')

        def test_get_text(self):
            backend = PetrelBackend()
            text_path = f'{self.petrel_dir}/text1.txt'
            self.assertEqual(backend.get_text(text_path), 'text1')

        def test_put(self):
            backend = PetrelBackend()
            img_path = f'{self.petrel_dir}/img.jpg'
            backend.put(b'img', img_path)

        def test_put_text(self):
            backend = PetrelBackend()
            text_path = f'{self.petrel_dir}/text5.txt'
            backend.put_text('text5', text_path)

        def test_exists(self):
            backend = PetrelBackend()

            # file and directory exist
            dir_path = f'{self.petrel_dir}/dir2'
            self.assertTrue(backend.exists(dir_path))
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertTrue(backend.exists(img_path))

            # file and directory does not exist
            not_existed_dir = f'{self.petrel_dir}/not_existed_dir'
            self.assertFalse(backend.exists(not_existed_dir))
            not_existed_path = f'{self.petrel_dir}/img.jpg'
            self.assertFalse(backend.exists(not_existed_path))

        def test_isdir(self):
            backend = PetrelBackend()
            dir_path = f'{self.petrel_dir}/dir2'
            self.assertTrue(backend.isdir(dir_path))
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertFalse(backend.isdir(img_path))

        def test_isfile(self):
            backend = PetrelBackend()
            dir_path = f'{self.petrel_dir}/dir2'
            self.assertFalse(backend.isfile(dir_path))
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertTrue(backend.isfile(img_path))

        def test_get_local_path(self):
            backend = PetrelBackend()
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            with backend.get_local_path(img_path) as path:
                self.assertTrue(osp.isfile(path))
                self.assertEqual(Path(path).open('rb').read(), b'img')
            # exist the with block and path will be released
            self.assertFalse(osp.isfile(path))

        def test_copyfile(self):
            backend = PetrelBackend()

            # dst is a file
            src = f'{self.petrel_dir}/dir2/img.jpg'
            dst = f'{self.petrel_dir}/img.jpg'
            self.assertEqual(backend.copyfile(src, dst), dst)
            self.assertTrue(backend.isfile(dst))

            # dst is a directory
            dst = f'{self.petrel_dir}/dir1'
            expected_dst = f'{self.petrel_dir}/dir1/img.jpg'
            self.assertEqual(backend.copyfile(src, dst), expected_dst)
            self.assertTrue(backend.isfile(expected_dst))

            # src and src should not be same file
            with self.assertRaises(SameFileError):
                backend.copyfile(src, src)

        def test_copytree(self):
            backend = PetrelBackend()
            src = f'{self.petrel_dir}/dir2'
            dst = f'{self.petrel_dir}/dir3'
            self.assertFalse(backend.exists(dst))
            self.assertEqual(backend.copytree(src, dst), dst)
            self.assertEqual(
                list(backend.list_dir_or_file(src)),
                list(backend.list_dir_or_file(dst)))

            # dst should not exist
            with self.assertRaises(FileExistsError):
                backend.copytree(src, dst)

        def test_copyfile_from_local(self):
            backend = PetrelBackend()

            # dst is a file
            src = self.local_img_path
            dst = f'{self.petrel_dir}/color.jpg'
            self.assertFalse(backend.exists(dst))
            self.assertEqual(backend.copyfile_from_local(src, dst), dst)
            self.assertTrue(backend.isfile(dst))

            # dst is a directory
            src = self.local_img_path
            dst = f'{self.petrel_dir}/dir1'
            expected_dst = f'{self.petrel_dir}/dir1/color.jpg'
            self.assertFalse(backend.exists(expected_dst))
            self.assertEqual(
                backend.copyfile_from_local(src, dst), expected_dst)
            self.assertTrue(backend.isfile(expected_dst))

        def test_copytree_from_local(self):
            backend = PetrelBackend()
            backend.rmtree(self.petrel_dir)
            with build_temporary_directory() as tmp_dir:
                backend.copytree_from_local(tmp_dir, self.petrel_dir)
                files = backend.list_dir_or_file(
                    self.petrel_dir, recursive=True)
                self.assertEqual(len(list(files)), 8)

        def test_copyfile_to_local(self):
            backend = PetrelBackend()
            with tempfile.TemporaryDirectory() as tmp_dir:
                # dst is a file
                src = f'{self.petrel_dir}/dir2/img.jpg'
                dst = Path(tmp_dir) / 'img.jpg'
                self.assertEqual(backend.copyfile_to_local(src, dst), dst)
                self.assertEqual(dst.open('rb').read(), b'img')

                # dst is a directory
                dst = Path(tmp_dir) / 'dir'
                dst.mkdir()
                self.assertEqual(
                    backend.copyfile_to_local(src, dst), dst / 'img.jpg')
                self.assertEqual((dst / 'img.jpg').open('rb').read(), b'img')

        def test_copytree_to_local(self):
            backend = PetrelBackend()
            with tempfile.TemporaryDirectory() as tmp_dir:
                backend.copytree_to_local(self.petrel_dir, tmp_dir)
                self.assertTrue(osp.exists(Path(tmp_dir) / 'text1.txt'))
                self.assertTrue(osp.exists(Path(tmp_dir) / 'dir2' / 'img.jpg'))

        def test_remove(self):
            backend = PetrelBackend()
            img_path = f'{self.petrel_dir}/dir2/img.jpg'
            self.assertTrue(backend.isfile(img_path))
            backend.remove(img_path)
            self.assertFalse(backend.exists(img_path))

        def test_rmtree(self):
            backend = PetrelBackend()
            dir_path = f'{self.petrel_dir}/dir2'
            self.assertTrue(backend.isdir(dir_path))
            backend.rmtree(dir_path)
            self.assertFalse(backend.exists(dir_path))

        def test_copy_if_symlink_fails(self):
            backend = PetrelBackend()

            # dst is a file
            src = f'{self.petrel_dir}/dir2/img.jpg'
            dst = f'{self.petrel_dir}/img.jpg'
            self.assertFalse(backend.exists(dst))
            self.assertFalse(backend.copy_if_symlink_fails(src, dst))
            self.assertTrue(backend.isfile(dst))

            # dst is a directory
            src = f'{self.petrel_dir}/dir2'
            dst = f'{self.petrel_dir}/dir'
            self.assertFalse(backend.exists(dst))
            self.assertFalse(backend.copy_if_symlink_fails(src, dst))
            self.assertTrue(backend.isdir(dst))

        def test_list_dir_or_file(self):
            backend = PetrelBackend()

            # list directories and files
            self.assertEqual(
                set(backend.list_dir_or_file(self.petrel_dir)),
                {'dir1', 'dir2', 'text1.txt', 'text2.txt'})

            # list directories and files recursively
            self.assertEqual(
                set(backend.list_dir_or_file(self.petrel_dir, recursive=True)),
                {
                    'dir1', '/'.join(('dir1', 'text3.txt')), 'dir2', '/'.join(
                        ('dir2', 'dir3')), '/'.join(
                            ('dir2', 'dir3', 'text4.txt')), '/'.join(
                                ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                })

            # only list directories
            self.assertEqual(
                set(
                    backend.list_dir_or_file(self.petrel_dir,
                                             list_file=False)),
                {'dir1', 'dir2'})
            with self.assertRaisesRegex(
                    TypeError,
                    '`list_dir` should be False when `suffix` is not None'):
                backend.list_dir_or_file(
                    self.petrel_dir, list_file=False, suffix='.txt')

            # only list directories recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir, list_file=False, recursive=True)),
                {'dir1', 'dir2', '/'.join(('dir2', 'dir3'))})

            # only list files
            self.assertEqual(
                set(backend.list_dir_or_file(self.petrel_dir, list_dir=False)),
                {'text1.txt', 'text2.txt'})

            # only list files recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir, list_dir=False, recursive=True)),
                {
                    '/'.join(('dir1', 'text3.txt')), '/'.join(
                        ('dir2', 'dir3', 'text4.txt')), '/'.join(
                            ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                })

            # only list files ending with suffix
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir, list_dir=False, suffix='.txt')),
                {'text1.txt', 'text2.txt'})
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir,
                        list_dir=False,
                        suffix=('.txt', '.jpg'))), {'text1.txt', 'text2.txt'})
            with self.assertRaisesRegex(
                    TypeError,
                    '`suffix` must be a string or tuple of strings'):
                backend.list_dir_or_file(
                    self.petrel_dir, list_dir=False, suffix=['.txt', '.jpg'])

            # only list files ending with suffix recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir,
                        list_dir=False,
                        suffix='.txt',
                        recursive=True)), {
                            '/'.join(('dir1', 'text3.txt')), '/'.join(
                                ('dir2', 'dir3', 'text4.txt')), 'text1.txt',
                            'text2.txt'
                        })

            # only list files ending with suffix
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        self.petrel_dir,
                        list_dir=False,
                        suffix=('.txt', '.jpg'),
                        recursive=True)),
                {
                    '/'.join(('dir1', 'text3.txt')), '/'.join(
                        ('dir2', 'dir3', 'text4.txt')), '/'.join(
                            ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                })
