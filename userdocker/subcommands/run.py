# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import re
import subprocess
import shlex

from .. import __version__
from ..config import ALLOWED_IMAGE_REGEXPS
from ..config import ALLOWED_PORT_MAPPINGS
from ..config import CAPS_ADD
from ..config import CAPS_DROP
from ..config import ENV_VARS
from ..config import ENV_VARS_EXT
from ..config import NUM_CPUS_DEFAULT
from ..config import NV_ALLOW_OWN_GPU_REUSE
from ..config import NV_ALLOWED_GPUS
from ..config import NV_DEFAULT_GPU_COUNT_RESERVATION
from ..config import NV_MAX_GPU_COUNT_RESERVATION
from ..config import PROBE_USED_MOUNTS
from ..config import RUN_PULL
from ..config import USER_IN_CONTAINER
from ..config import VOLUME_MOUNTS_ALWAYS
from ..config import VOLUME_MOUNTS_AVAILABLE
from ..config import VOLUME_MOUNTS_DEFAULT
from ..config import gid
from ..config import uid
from ..config import user_name
from ..helpers.cmd import init_cmd
from ..helpers.container import container_get_next_name
from ..helpers.exceptions import UserDockerException
from ..helpers.execute import exec_cmd
from ..helpers.execute import exit_exec_cmd
from ..helpers.logger import logger
from ..helpers.nvidia import nvidia_get_available_gpus
from ..helpers.parser import init_subcommand_parser
from ..helpers.parser import split_into_arg_and_value


def parser_run(parser):
    """Create parser for the run subcommand.

    See the config file for details on the available arguments and options.
    """
    sub_parser = init_subcommand_parser(parser, 'run')

    sub_parser.add_argument(
        "--no-default-mounts",
        help="does not automatically add default mounts",
        action="store_true",
    )

    mounts_help = []
    if VOLUME_MOUNTS_ALWAYS:
        mounts_help += ['Admin enforced: %s.' % ', '.join(VOLUME_MOUNTS_ALWAYS)]
    if VOLUME_MOUNTS_DEFAULT:
        mounts_help += ['Default: %s.' % ', '.join(VOLUME_MOUNTS_DEFAULT)]
    if VOLUME_MOUNTS_AVAILABLE:
        mounts_help += ['Available: %s.' % ', '.join(VOLUME_MOUNTS_AVAILABLE)]
    if mounts_help:
        sub_parser.add_argument(
            "-v", "--volume",
            help="user specified volume mounts (can be given multiple times). "
                 "%s" % " ".join(mounts_help),
            action="append",
            dest="volumes",
            default=[],
        )

    if ALLOWED_PORT_MAPPINGS:
        sub_parser.add_argument(
            "-p", "--publish",
            help="Publish a container's ports to the host (see docker help). "
                 "Allowed: " + ', '.join(ALLOWED_PORT_MAPPINGS),
            action="append",
            dest="port_mappings",
            default=[],
        )

    sub_parser.add_argument(
        "--name",
        help="container name. Your username plus a number by default.",
    )

    sub_parser.add_argument(
        "image",
        help="the image to run. Allowed: " + ', '.join(ALLOWED_IMAGE_REGEXPS),
    )

    sub_parser.add_argument(
        "image_args",
        help="arguments passed to the image",
        nargs=argparse.REMAINDER
    )


def prepare_nvidia_docker_run(args):
    """Prepare the execution of the run subcommand for nvidia-docker.

    Mainly handles GPU arbitration via ENV var for nvidia-docker. Note that
    these are ENV vars for the command, not the container.
    """

    if os.getenv('NV_HOST'):
        raise UserDockerException('ERROR: NV_HOST env var not supported yet')

    if os.getenv('NV_GPU'):
        raise UserDockerException('ERROR: NV_GPU no longer supported, use NVIDIA_VISIBLE_DEVICES instead')

    # check if allowed
    if not NV_ALLOWED_GPUS:
        raise UserDockerException(
            "ERROR: No GPUs available due to admin setting."
        )

    nv_visible_devices = os.getenv('NVIDIA_VISIBLE_DEVICES', '')
    if nv_visible_devices:
        # the user has set NVIDIA_VISIBLE_DEVICES, just check if it's ok
        nv_visible_devices = [g.strip() for g in nv_visible_devices.split(',')]

        if not (
                NV_ALLOWED_GPUS == 'ALL'
                or all(gpu in NV_ALLOWED_GPUS for gpu in nv_visible_devices)):
            raise UserDockerException(
                "ERROR: Access to at least one specified NVIDIA_VISIBLE_DEVICES denied by "
                "admin. Available GPUs: %r" % (NV_ALLOWED_GPUS,)
            )

        # check if in bounds (and MAX >= 0)
        if 0 <= NV_MAX_GPU_COUNT_RESERVATION < len(nv_visible_devices):
            raise UserDockerException(
                "ERROR: Number of requested GPUs > %d (admin limit)" % (
                    NV_MAX_GPU_COUNT_RESERVATION,)
            )

        # check if available
        gpus_available, own_gpus = nvidia_get_available_gpus(args.executor_path)
        if NV_ALLOW_OWN_GPU_REUSE:
            gpus_available.extend(own_gpus)
        for g in nv_visible_devices:
            if g not in gpus_available:
                msg = (
                    'ERROR: GPU %s is currently not available!\nUse:\n'
                    '"sudo userdocker ps --gpu-free" to find available GPUs.\n'
                    '"sudo userdocker ps --gpu-used" and "nvidia-smi" to see '
                    'status.' % g
                )
                if NV_ALLOW_OWN_GPU_REUSE and own_gpus:
                    msg += '\n"sudo userdocker ps --gpu-used-mine to show own' \
                           '(reusable) GPUs.'
                raise UserDockerException(msg)
    else:
        # NVIDIA_VISIBLE_DEVICES wasn't set, use admin defaults, tell user
        gpu_default = NV_DEFAULT_GPU_COUNT_RESERVATION
        logger.info(
            "NVIDIA_VISIBLE_DEVICES environment variable not set, trying to acquire admin "
            "default of %d GPUs" % gpu_default
        )
        gpus_available, own_gpus = nvidia_get_available_gpus(args.executor_path)
        gpus = gpus_available[:gpu_default]
        if len(gpus) < gpu_default:
            msg = (
                'Could not find %d available GPU(s)!\nUse:\n'
                '"sudo userdocker ps --gpu-used" and "nvidia-smi" to see '
                'status.' % gpu_default
            )
            if NV_ALLOW_OWN_GPU_REUSE and own_gpus:
                msg += '\n You can set NVIDIA_VISIBLE_DEVICES to reuse a GPU you have already' \
                       ' reserved.'
            raise UserDockerException(msg)
        gpu_env = ",".join([str(g) for g in gpus])
        logger.info("Setting NVIDIA_VISIBLE_DEVICES=%s" % gpu_env)
        os.environ['NVIDIA_VISIBLE_DEVICES'] = gpu_env


def determine_cpu_params(args, cmd):
    num_cpus = open('/proc/cpuinfo').read().count('processor\t:')

    # No default number of CPUs and none specified by the user: nothing to do
    if NUM_CPUS_DEFAULT == 0 and (not hasattr(args, 'cpus') or not args.cpus) \
            and all([not s.startswith("--cpus=") for s in args.patch_through_args]):
        return

    pargs = args.patch_through_args

    passthrough_cpus_value = None
    if any([s.startswith("--cpus=") for s in pargs]):
        cpu_arg = pargs[[i for i in range(len(pargs))
                         if pargs[i].startswith("--cpus=")][0]]
        passthrough_cpus_value = split_into_arg_and_value(cpu_arg)[1]

    passthrough_cpuset_cpus_value = None
    if any([s.startswith("--cpuset-cpus=") for s in pargs]):
        cpuset_cpus_arg = pargs[[i for i in range(len(pargs)) if
                            pargs[i].startswith("--cpuset-cpus=")][0]]
        passthrough_cpuset_cpus_value = split_into_arg_and_value(cpuset_cpus_arg)[1]

    # Add --cpus if not given
    if not hasattr(args, 'cpus') or not args.cpus:
        args.cpus = int(passthrough_cpus_value) if passthrough_cpus_value else int(NUM_CPUS_DEFAULT)
        if not passthrough_cpus_value:
            cmd += ['--cpus', str(args.cpus)]

    # Add --cpuset_cpus if not given
    if not hasattr(args, 'cpuset-cpus') or not args.cpuset_cpus:
        if not passthrough_cpuset_cpus_value:
            cpu_indices = [str(x) for x in range(num_cpus)]
            random.shuffle(cpu_indices)
            args.cpuset_cpus = ",".join(cpu_indices[0:args.cpus])
            cmd += ['--cpuset-cpus', str(args.cpuset_cpus)]


def exec_cmd_run(args):
    """Execute the run subcommand."""
    cmd = init_cmd(args)

    # check port mappings
    for pm in getattr(args, 'port_mappings', []):
        for pm_pattern in ALLOWED_PORT_MAPPINGS:
            if re.match(pm_pattern, pm):
                cmd += ['-p', pm]
                break
        else:
            raise UserDockerException(
                "ERROR: given port mapping not allowed: %s" % pm
            )

    mounts = []
    mounts_available = \
        VOLUME_MOUNTS_ALWAYS + VOLUME_MOUNTS_DEFAULT + VOLUME_MOUNTS_AVAILABLE

    mounts += VOLUME_MOUNTS_ALWAYS

    if not args.no_default_mounts:
        mounts += VOLUME_MOUNTS_DEFAULT

    for user_mount in getattr(args, 'volumes', []):
        if user_mount in mounts:
            continue
        if user_mount in mounts_available:
            mounts += [user_mount]
            continue

        # literal matches didn't work, check if the user appended a 'ro' flag
        if len(user_mount.split(':')) == 3:
            host_path, container_path, flag = user_mount.split(':')
            if flag == 'ro':
                st = ':'.join([host_path, container_path])
                if st in mounts:
                    # upgrade mount to include ro flag
                    idx = mounts.index(st)
                    mounts[idx] = user_mount
                    continue
                if st in mounts_available:
                    mounts += [user_mount]
                    continue

        # allow potentially unspecified container_path mounts
        host_path = user_mount.split(':')[0] + ':'
        if host_path in mounts:
            continue

        raise UserDockerException(
            "ERROR: given mount not allowed: %s" % user_mount
        )

    mount_host_paths = [m.split(':')[0] for m in mounts]
    for ms in mount_host_paths:
        if not os.path.exists(ms):
            logger.warn("Mount {0} does not exist, trying to create it."
                        .format(ms))
            if subprocess.call(["/usr/bin/mkdir", "-p", ms]) != 0:
                raise UserDockerException(
                    "ERROR: mount could not be created: {0}".format(ms)
                )
        if PROBE_USED_MOUNTS and os.path.isdir(ms):
            os.listdir(ms)

    for mount in mounts:
        if ':' not in mount:
            raise UserDockerException(
                "ERROR: anonymous mounts currently not supported: %s" % mount
            )
        cmd += ["-v", mount]

    if args.executor == 'nvidia-docker':
        prepare_nvidia_docker_run(args)

    env_vars = ENV_VARS + ENV_VARS_EXT.get(args.executor, [])
    env_vars += [
        "USERDOCKER=%s" % __version__,
        "USERDOCKER_USER=%s" % user_name,
        "USERDOCKER_UID=%d" % uid,
    ]
    if args.executor == 'nvidia-docker':
        # remember which GPU was assigned to the container for ps --gpu-used
        env_vars += [
            "USERDOCKER_NV_VISIBLE_DEVICES=%s" % os.environ['NVIDIA_VISIBLE_DEVICES'],
            "NVIDIA_VISIBLE_DEVICES=%s" % os.environ['NVIDIA_VISIBLE_DEVICES']
        ]
        cmd += ["--runtime=nvidia"]
    for env_var in env_vars:
        cmd += ['-e', env_var]


    if USER_IN_CONTAINER:
        cmd += ["-u", "%d:%d" % (uid, gid)]

    for cap_drop in CAPS_DROP:
        cmd += ["--cap-drop=%s" % cap_drop]
    for cap_add in CAPS_ADD:
        cmd += ["--cap-add=%s" % cap_add]

    # add default container name if not specified by the user
    name = container_get_next_name(args.executor_path)\
        if not args.name else args.name

    cmd += ['--name', name]

    determine_cpu_params(args, cmd)

    # additional injection protection, deactivated for now due to nvidia-docker
    # inability to handle this
    # cmd.append("--")

    img = args.image
    if ":" not in img and "@" not in img:
        # user didn't explicitly set a tag or digest, append ":latest"
        img += ":latest"

    if ALLOWED_IMAGE_REGEXPS:
        for air in ALLOWED_IMAGE_REGEXPS:
            if re.match(air, img):
                break
        else:
            raise UserDockerException(
                "ERROR: image %s not in allowed image regexps: %s" % (
                    img, ALLOWED_IMAGE_REGEXPS))

    # pull image?
    if RUN_PULL == "default":
        # just let `docker run` do its thing
        pass
    elif RUN_PULL == "always":
        # pull image
        exec_cmd(
            [args.executor_path, 'pull', img],
            dry_run=args.dry_run,
            loglvl=logging.DEBUG,
        )
    elif RUN_PULL == "never":
        # check if image is available locally
        tmp = exec_cmd(
            [args.executor_path, 'images', '-q', img],
            return_status=False,
            loglvl=logging.DEBUG,
        )
        if not tmp:
            raise UserDockerException(
                "ERROR: you can only use locally available images, but %s could"
                " not be found locally" % img
            )
    else:
        raise UserDockerException(
            "ERROR: RUN_PULL config variable not expected range, contact admin"
        )

    cmd.append(img)
    cmd.extend(args.image_args)

    exit_exec_cmd(cmd, dry_run=args.dry_run)
