# A helper file for data processing functions
from unidecode import unidecode
def formatNames(playernames):
    
    # Fix a few dumb ones
    playernames = ["Jeffery Taylor" if name=="Jeff Taylor" else name for name in playernames]
    playernames= ["Gustavo Alfonso Ayon" if name=="Gustavo Ayón" else name for name in playernames]
    
    playernames = [name.replace(" III", "") for name in playernames]
    playernames = [name.replace(" II", "") for name in playernames]
    #playernames = [name.replace(" I", "") for name in playernames]
    playernames = [name.replace("Jr.", "") for name in playernames]
    playernames = [name.replace("Sr.", "") for name in playernames]
    playernames = [name.replace("'", "") for name in playernames]
    playernames = [name.replace(".", "") for name in playernames]
    playernames = [name.replace("-", "") for name in playernames]
    playernames = [name.replace(" ", "") for name in playernames]
    
    playernames = [unidecode(name) for name in playernames]

    playernames = [name.lower() for name in playernames]
    return playernames