from pathlib import Path

def dataDirectory(dataDirectoryName='data'):
    """
    Zoek omhoog in de mapstructuur tot een directory met de naam 'data' wordt gevonden.
    Retourneert het pad naar die directory.
    """
    # Start in de map waar dit bestand zich bevindt
    dataDir = Path(__file__).resolve().parent

    # Loop omhoog totdat we een 'data'-map vinden of aan de root zijn
    while dataDir != dataDir.root:
        candidate = dataDir / dataDirectoryName
        if candidate.is_dir():
            return candidate
        dataDir = dataDir.parent  # één niveau omhoog

    # Geen data-directory gevonden
    raise FileNotFoundError(
        f"❌ Geen map '{dataDirectoryName}' gevonden in bovenliggende directories van {__file__}"
    )
