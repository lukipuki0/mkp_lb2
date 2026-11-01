"""Estilos nativos de draw.io por alias: usa d.vertex(x, y, w, h, nombre, style('alias'))."""
AZ = 'aspect=fixed;html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;image;image=img/lib/azure2/'
SH = 'html=1;labelPosition=center;verticalLabelPosition=bottom;align=center;verticalAlign=top;shape='
S = {  # alias: estilo nativo; nombre oficial del producto en el comentario
    'app-service': AZ + 'app_services/App_Services.svg',              # Azure App Service
    'postgres': AZ + 'databases/Azure_Database_PostgreSQL_Server.svg',  # Azure Database for PostgreSQL
    'key-vault': AZ + 'security/Key_Vaults.svg',                        # Azure Key Vault
    'monitor': AZ + 'management_governance/Monitor.svg',                # Azure Monitor
    'private-link': AZ + 'networking/Private_Link.svg',                 # Azure Private Link
    'vnet': AZ + 'networking/Virtual_Networks.svg',                     # Azure Virtual Network
    'storage': AZ + 'storage/Storage_Accounts.svg',                     # Azure Storage
    'fortigate': SH + 'mxgraph.fortinet.fortigate;fillColor=#EE3124;strokeColor=none;',
    'switch': SH + 'mxgraph.cisco.switches.workgroup_switch;fillColor=#036897;strokeColor=#ffffff;',
    'router': SH + 'mxgraph.cisco.routers.router;fillColor=#036897;strokeColor=#ffffff;',
    'server': SH + 'mxgraph.rack.general.1u_rack_server;',
    'ups': SH + 'mxgraph.rack.apc.apc_smart_ups_1000-3000_va_2u;',
    'patch-panel': SH + 'mxgraph.rack.general.cat5e_rack_mount_patch_panel_24_ports;',
}


def style(alias):
    """Devuelve el estilo nativo; falla si el alias no existe para no inventar íconos."""
    return S[alias]
