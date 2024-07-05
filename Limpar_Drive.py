from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
gauth = GoogleAuth()
scope = ['https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('conta-de-servico.json', scope)
gauth.credentials = creds

# Criação do objeto drive
drive = GoogleDrive(gauth)
file_list = drive.ListFile({'q': f"trashed=false"}).GetList()
for file in file_list:
    file.Delete()
