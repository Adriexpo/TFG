import csv
import difflib

def find_closest_set(extracted_text):
    # Define the mapping between predicted set IDs and CSV file names
    set_mapping = {
        "RDS-SP": "AUGE DEL DESTINO",
        "FET-SP": "ETERNIDAD LLAMEANTE",
        "LON-": "LABYRINTH OF NIGHTMARE",
        "STON-SP": "GOLPE DE NEOS",
        "FOTB-SP": "FUERZA DEL ROMPEDOR",
        "EOJ-SP": "ENEMIGO DE LA JUSTICIA",
        "DR3-SP": "REVELACIÓN OSCURA VOLUMEN 3",
        "CRV-SP": "REVOLUCIÓN CIBERNÉTICA",
        "DR1-SP": "REVELACIÓN OSCURA VOLUMEN 1",
        "POTD-SP": "PODER DEL DUELISTA",
        "SOD-SP": "ALMA DEL DUELISTA",
        "PTDN-SP": "OSCURIDAD FANTASMA",
        "YSD-SP": "STARTER DECK 2006",
        "TLM-SP": "EL MILENIO PERDIDO",
        "SDH-SP": "SEÑOR DE LOS HECHIZOS",
        "MFC-": "MAGICIAN'S FORCE",
        "PMT-S": "PREDADORES METÁLICOS",
        "IOC-SP": "INVASIÓN DEL CAOS",
        "AST-SP": "SANTUARIO ANTIGUO",
        "LDD-S": "LEYENDA DEL DRAGÓN BLANCO DE OJOS AZULES",
        "LOB-S": "LEYENDA DEL DRAGÓN BLANCO DE OJOS AZULES",
        "BIJ-S": "BARAJA DE INICIO JOEY",
        "BIP-S": "BARAJA DE INICIO PEGASUS",
        "GLD2-SP": "SERIE DORADA 2009",
        "LODT-SP": "LUZ DE LA DESTRUCCIÓN",
        "DP03-SP": "SOBRE DE DUELISTA - JADEN YUKI 3",
        "SD09-SP": "BARAJA DE ESTRUCTURA CÓLERA DE DINOSAURIO",
        "DP04-SP": "SOBRE DE DUELISTA - ZANE TRUESDALE",
        "DB1-SP": "PRINCIPIO OSCURO 1"
    }

    # Load set names from CSV
    sets = []
    with open('set_id.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            sets.extend(row)
    
    # Compare extracted text with set names
    similarities = {}
    for set_name in sets:
        similarity = difflib.SequenceMatcher(None, extracted_text, set_name).ratio()
        similarities[set_name] = similarity
    
    # Find the set with the highest similarity
    closest_set = max(similarities, key=similarities.get)
    
    # Get the related CSV file for the predicted set
    related_csv_file = set_mapping.get(closest_set)
    
    return related_csv_file
