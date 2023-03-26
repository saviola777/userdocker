# -*- coding: utf-8 -*-

import json
import logging
import re
from collections import defaultdict
from typing import Tuple
from xml.etree import ElementTree as ET

from ..config import uid, NVIDIA_SMI, NV_ALLOWED_GPUS, \
    NV_EXCLUSIVE_CONTAINER_GPU_RESERVATION, \
    NV_GPU_UNAVAILABLE_ABOVE_MEMORY_USED
from .logger import logger
from .execute import exec_cmd
from .container import container_get_running


def get_gpu_and_mig_uuids(nvidia_smi) -> dict:
    """Extracts the GPU and MIG UUIDs from nvidia-smi -L

    Returns:
        ``dict`` mapping GPU UUIDs to a list of MIG UUIDs
    """
    gpu_list_str = exec_cmd(
        [nvidia_smi,
         '--list-gpus',],
        return_status=False,
        loglvl=logging.DEBUG,
    )

    gpu_uuid = None
    mig_uuids = {}
    for line in gpu_list_str.split("\n"):
        result = re.search("(GPU-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", line)
        if result is not None:
            gpu_uuid = result.group(1)
            mig_uuids[gpu_uuid] = []

        result = re.search(
            "(MIG-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", line)
        if result is not None:
            mig_uuids[gpu_uuid].append(result.group(1))

    return mig_uuids


def extract_gis_with_memory(nvidia_smi, mig_uuids) -> dict:
    """Returns a dictionary mapping GPU instance UUIDs to memory usage
    """

    gis = {}

    for gpu_uuid in mig_uuids.keys():
        gpu_info_str = exec_cmd(
            [nvidia_smi,
             '--query',
             '--id=' + gpu_uuid,
             '--xml-format'],
            return_status=False,
            loglvl=logging.DEBUG,
        )

        root = ET.fromstring(gpu_info_str)

        gpu = root.find('gpu')
        uuid = gpu.find("uuid").text

        mig_enabled = gpu.find(".//current_mig").text == "Enabled"

        if mig_enabled:
            mig_index = 0
            for mig in gpu.find("mig_devices").findall("mig_device"):
                uuid = mig_uuids[gpu_uuid][mig_index]

                gis[uuid] = int(mig.find("fb_memory_usage/used").text[:-4])

                mig_index += 1
        else:
            gis[uuid] = int(gpu.find("fb_memory_usage/used").text[:-4])

    return gis


def container_find_userdocker_user_uid_gpus(container_env):
    pairs = [var.partition('=') for var in container_env]
    users = [v for k, _, v in pairs if k == 'USERDOCKER_USER']
    uids = [v for k, _, v in pairs if k == 'USERDOCKER_UID']
    gpus = [v for k, _, v in pairs if k == 'USERDOCKER_NV_VISIBLE_DEVICES']
    if gpus:
        gpus = gpus[0].split(',')
    return users[0] if users else '', uids[0] if uids else None, gpus


def nvidia_get_gpus_used_by_containers(docker: str) -> defaultdict:
    """Return the GPUs currently used by docker containers.

    Args:
        docker: Path of the docker executor.

    Returns:
        ``defaultdict`` mapping GPU IDs to a list
            ``(container, container_name, container_user, container_uid)``
    """

    running_containers = container_get_running(docker)

    gpu_used_by_containers = defaultdict(list)

    if not running_containers:
        return gpu_used_by_containers

    gpu_used_by_containers_str = exec_cmd(
        [
            docker, 'inspect', '--format',
            '[{{json .Name}}, {{json .Id}}, {{json .Config.Env}}]'
        ] + running_containers,
        return_status=False,
        loglvl=logging.DEBUG,
    )
    logger.debug('gpu_used_by_containers_str: %s', gpu_used_by_containers_str)

    for line in gpu_used_by_containers_str.splitlines():
        container_name, container, container_env = json.loads(line)
        container_user, container_uid, gpus = \
            container_find_userdocker_user_uid_gpus(container_env)
        for gpu_uuid in gpus:
            gpu_used_by_containers[gpu_uuid].append(
                (container, container_name, container_user, container_uid)
            )
            logger.debug(
                'gpu %d used by container: %s, name: %s, user: %s, uid: %s',
                gpu_uuid, container, container_name, container_user, container_uid
            )
    return gpu_used_by_containers


def nvidia_get_available_gpus(docker: str, nvidia_smi: str=NVIDIA_SMI) -> Tuple[list, list]:
    """Return the available GPUs.

    Availability of GPUs depends on:
    - ``NV_ALLOWED_GPUS``: GPUs not in this list are generally not available
    - ``NV_GPU_UNAVAILABLE_ABOVE_MEMORY_USED``: if more than this amount of memory
            is used, a GPU is considered unavailable
    - ``NV_EXCLUSIVE_CONTAINER_GPU_RESERVATION``: if ``True``, GPUs used by other
            containers are considered unavailable

    See the configuration file for additional details.
    """
    if not NV_ALLOWED_GPUS:
        return list(), list()

    gpu_mem_used = extract_gis_with_memory(nvidia_smi, get_gpu_and_mig_uuids(nvidia_smi))
    logger.debug('gpu usage:\n%s', gpu_mem_used)

    gpus_used_by_containers = nvidia_get_gpus_used_by_containers(docker)
    gpus_used_by_own_containers = [
        gpu for gpu, info in gpus_used_by_containers.items()
        if any(i[3] == uid for i in info)
    ]

    # get available gpus asc by mem used and reservation counts
    mem_limit = NV_GPU_UNAVAILABLE_ABOVE_MEMORY_USED
    mem_res_gpu = [
        (m, len(gpus_used_by_containers.get(gpu, [])), gpu)
        for gpu, m in gpu_mem_used.items()
    ]
    available_gpus = [
        g for m, r, g in sorted(mem_res_gpu) if mem_limit < 0 or m <= mem_limit
    ]
    if NV_ALLOWED_GPUS != 'ALL':
        available_gpus = [g for g in available_gpus if g in NV_ALLOWED_GPUS]
    logger.debug(
        'available GPUs after mem and allowance filtering: %r', available_gpus)

    if NV_EXCLUSIVE_CONTAINER_GPU_RESERVATION:
        available_gpus = [
            gpu for gpu in available_gpus
            if gpu not in gpus_used_by_containers
        ]

    return available_gpus, gpus_used_by_own_containers
