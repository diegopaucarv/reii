"""reii.pipelines — mapa de pipelines, sincronización con la BD y pestaña de configuración.

Orden de lectura: registry -> discourse_db -> config_sync -> status ->
discourse_results (final de la pestaña C) -> hub_tab (última pestaña).
"""
