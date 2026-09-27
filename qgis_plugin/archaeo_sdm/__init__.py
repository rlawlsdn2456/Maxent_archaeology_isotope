# -*- coding: utf-8 -*-
"""Archaeo-SDM QGIS 플러그인 진입점."""


def classFactory(iface):
    from .plugin import ArchaeoSdmPlugin
    return ArchaeoSdmPlugin(iface)
