# -*- coding: utf-8 -*-

import argparse

from ..config import ARGS_AVAILABLE, ARGS_ALWAYS
from .cmd import init_cmd


def arg_type_no_flag(string):
    if string.startswith('-'):
        raise argparse.ArgumentTypeError('%r cannot start with "-"' % string)
    return string


def init_subcommand_parser(parent_parser, scmd):
    parser = parent_parser.add_parser(
        scmd,
        help='Lets a user run "docker %s ..." command' % scmd
    )
    parser.set_defaults(
        patch_through_args=[],
    )

    # patch args through
    _args_seen = []
    for args in ARGS_AVAILABLE.get(scmd, []) + ARGS_ALWAYS.get(scmd, []):
        if isinstance(args, str):
            # just a single arg as string
            args = [args]
        elif isinstance(args, (list, tuple)):
            # aliases as list or tuple
            args = list(args)
        else:
            raise NotImplementedError(
                "Cannot understand admin defined ARG %s for command %s" % (
                    args, scmd))

        # remove dups (e.g. from being in AVAILABLE and ALWAYS)
        args = [arg for arg in args if arg not in _args_seen]
        _args_seen.extend(args)

        # make sure arg starts with - and doesn't contain = or ' '
        for arg in args:
            if not arg.startswith('-') or '=' in arg or ' ' in arg:
                raise NotImplementedError(
                    "Cannot understand admin defined ARG %s for command %s" % (
                        arg, scmd))

        h = "see docker help"
        if set(args) & set(ARGS_ALWAYS.get(scmd, [])):
            h += ' (enforced by admin)'
        parser.add_argument(
            *args,
            help=h,
            action="append_const",
            const=args[0],
            dest="patch_through_args",
        )

    return parser