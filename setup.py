import os
import re
from setuptools import find_packages, setup  # type: ignore

from pkg_resources import DistributionNotFound, get_distribution


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'mmengine/version.py'


def choose_requirement(primary, secondary):
    """If some version of primary requirement installed, return primary, else
    return secondary."""
    try:
        name = re.split(r'[!<>=]', primary)[0]
        get_distribution(name)
    except DistributionNotFound:
        return secondary

    return str(primary)


def get_version():
    vars = {}
    with open(version_file, encoding="utf-8") as f:
        exec(compile(f.read(), version_file, 'exec'), None, vars)
    return vars['__version__']


def parse_requirements(fname='requirements/runtime.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath) as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    yield from parse_line(line)

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if int(os.getenv('MMENGINE_LITE', '0')) == 1:
    install_requires = parse_requirements('requirements/runtime_lite.txt')
else:
    install_requires = parse_requirements()
    try:
        # OpenCV installed via conda.
        import cv2  # NOQA: F401
        major, minor, *rest = cv2.__version__.split('.')
        if int(major) < 3:
            raise RuntimeError(
                f'OpenCV >=3 is required but {cv2.__version__} is installed')
    except ImportError:
        # If first not installed install second package
        CHOOSE_INSTALL_REQUIRES = [('opencv-python-headless>=3',
                                    'opencv-python>=3')]
        for main, secondary in CHOOSE_INSTALL_REQUIRES:
            install_requires.append(choose_requirement(main, secondary))

setup(
    name='mmengine'
    if os.getenv('MMENGINE_LITE', '0') == '0' else 'mmengine-lite',
    version=get_version(),
    description='Engine of OpenMMLab projects',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/open-mmlab/mmengine',
    author='MMEngine Authors',
    author_email='openmmlab@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        'all': parse_requirements('requirements.txt'),
        'tests': parse_requirements('requirements/tests.txt'),
    },
)
