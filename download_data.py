import subprocess
import zipfile
import os
import shutil

# Télécharger les fichiers de la compétition Kaggle
def download_kaggle_data():
    if not os.path.exists('quora-question-pairs.zip'):  # Vérifie si le fichier zip est déjà téléchargé
        try:
            subprocess.run(["kaggle", "competitions", "download", "-c", "quora-question-pairs"], check=True)
            print("✅ Téléchargement terminé avec succès.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Une erreur est survenue lors du téléchargement : {e}")
    else:
        print("📂 'quora-question-pairs.zip' est déjà téléchargé.")

# Décompresser le fichier quora-question-pairs.zip dans le répertoire quora_data
def unzip_quora_zip():
    if os.path.exists('quora-question-pairs.zip'):
        if not os.path.exists('quora_data'):  # Vérifie si le dossier quora_data existe
            os.makedirs('quora_data')  # Crée le répertoire quora_data s'il n'existe pas
        if not os.path.exists('quora_data/train.csv.zip'):  # Vérifie si les fichiers sont déjà extraits
            with zipfile.ZipFile('quora-question-pairs.zip', 'r') as zip_ref:
                zip_ref.extractall('quora_data')  # Décompresse dans quora_data
            print("✅ Extraction de 'quora-question-pairs.zip' terminée dans 'quora_data'.")
        else:
            print("📂 Les fichiers sont déjà extraits dans 'quora_data'.")
    else:
        print("❌ Le fichier 'quora-question-pairs.zip' n'existe pas.")

# Décompresser les fichiers .zip (train.csv.zip, test.csv.zip, etc.) dans quora_data
def unzip_files():
    zip_file_paths = ['quora_data/train.csv.zip', 'quora_data/test.csv.zip', 'quora_data/sample_submission.csv.zip']
    
    for zip_file in zip_file_paths:
        if os.path.exists(zip_file):  # Vérifie si le fichier zip existe
            if not os.path.exists(zip_file.replace('.zip', '')):  # Vérifie si le fichier dézippé existe déjà
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall('quora_data')  # Décompresse dans quora_data
                print(f"✅ Extraction de {zip_file} terminée dans 'quora_data'.")
            else:
                print(f"📂 {zip_file.replace('.zip', '')} existe déjà.")
        else:
            print(f"❌ Le fichier {zip_file} n'existe pas.")

# Renommer le fichier test.csv en test2.csv dans le répertoire quora_data
def rename_test_file():
    if os.path.exists('quora_data/test.csv') and not os.path.exists('quora_data/test2.csv'):
        os.rename('quora_data/test.csv', 'quora_data/test2.csv')
        print("✅ Le fichier 'test.csv' a été renommé en 'test2.csv'.")
    elif os.path.exists('quora_data/test2.csv'):
        print("📂 'test2.csv' existe déjà.")
    else:
        print("❌ Le fichier 'test.csv' n'a pas été trouvé dans 'quora_data'.")

# Supprimer uniquement les fichiers ZIP après extraction, pas les fichiers CSV
def delete_zip_files():
    zip_file_paths = ['quora-question-pairs.zip', 'quora_data/train.csv.zip', 'quora_data/test.csv.zip', 'quora_data/sample_submission.csv.zip']

    # Supprimer les fichiers ZIP
    for zip_file in zip_file_paths:
        if os.path.exists(zip_file):
            os.remove(zip_file)
            print(f"✅ Le fichier {zip_file} a été supprimé.")
        else:
            print(f"📂 Le fichier {zip_file} n'existe pas ou a déjà été supprimé.")


# Fonction principale
def main():
    # Étape 1: Télécharger les données
    download_kaggle_data()

    # Étape 2: Décompresser le fichier quora-question-pairs.zip dans quora_data
    unzip_quora_zip()
    
    # Étape 3: Renommer le fichier test.csv en test2.csv
    rename_test_file()

    # Étape 4: Décompresser les fichiers .zip (train.csv.zip, test.csv.zip, etc.) dans quora_data
    unzip_files()

    # Étape 5: Supprimer les fichiers ZIP et les fichiers dézippés après extraction
    delete_zip_files()

# Appeler la fonction principale
if __name__ == "__main__":
    main()
