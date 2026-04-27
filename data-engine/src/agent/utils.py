
import os
import stat
from datetime import datetime

class File:
    """
    Get a file object, perform basic operations: open, rename, move, copy, delete, find newer/old version...
    """
    def __init__(self, path):

        self.path = path
        self.name = os.path.basename(path)

        if os.path.exists(path) is False:
            print(f'File [{self.name}] does not exist on this PC.')
            return
        
        if not os.path.isfile(self.path):
            self.is_file = False
            return
        else:
            self.is_file = True
            # print('This is a folder, not a file :(')
        
        self.location = os.path.dirname(path)
        self.user = self.get_user()
        self.owner = self.get_owner()

        self.last_modified_timestamp = os.path.getmtime(self.path)
        self.last_modified_time = datetime.fromtimestamp(self.last_modified_timestamp)
        self.creation_timestamp = os.path.getctime(self.path)
        self.creation_time = datetime.fromtimestamp(self.creation_timestamp)
        self.read_only = not os.access(self.path, os.W_OK)
        self.size_on_disk = os.path.getsize(self.path)


    def open(self):
        os.startfile(self.path)


    def rename(self, new_name):
        if os.path.exists(self.path + '\\' + new_name):
            print(f"File '{new_name}' already exists.")
            return
        os.rename(self.path, new_name)


    def get_user(self):
        import zipfile
        import xml.dom.minidom

        try:
            # Open the MS Office file to see the XML structure.
            document = zipfile.ZipFile(self.path)

            # Open/read the core.xml (contains the last user and modified date).
            uglyXML = xml.dom.minidom.parseString(document.read('docProps/core.xml')).toprettyxml(indent='  ')

            # Split lines in order to create a list.
            asText = uglyXML.splitlines()

            # loop the list in order to get the value you need. In my case last Modified By and the date.
            for item in asText:
                if 'lastModifiedBy' in item:
                    itemLength = len(item)-20
                    a_name = item[21:itemLength]
                    # print('Modified by:', item[21:itemLength])
                    info = 'Modified by:' + item[21:itemLength]

                if 'dcterms:modified' in item:
                    itemLength = len(item)-29
                    # print('Modified On:', item[46:itemLength])

            return info

        except:
            return 'unknown'
    

    def get_owner(self):
        import win32security
        sd = win32security.GetFileSecurity(self.path, win32security.OWNER_SECURITY_INFORMATION)
        owner_sid = sd.GetSecurityDescriptorOwner()
        owner_name, domain, _ = win32security.LookupAccountSid(None, owner_sid)
        return [owner_name, domain]


    def move_to(self, new_path, print_msg=True):

        if os.path.isdir(new_path):       
            new_path = new_path + '\\' + self.name

        if os.path.exists(new_path):
            print(f"File '{new_path}' already exists.")
            return

        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))

        # Move the file to the destination folder
        os.rename(self.path, new_path)

        if print_msg == True:
            print(f"File moved to {new_path}")

        self.location = os.path.dirname(new_path)
        self.path = new_path


    def delete(self):
        if os.path.exists(self.path):
            # Delete the file
            os.remove(self.path)
            print(f"File '{self.path}' deleted successfully.")
        else:
            print(f"File '{self.path}' does not exist.")


    def is_read_only(self):
        file_attributes = os.stat(self.path).st_mode
        # Check if user write permission is missing
        return not (file_attributes & stat.S_IWUSR)
    
    
    def set_read_only(self, is_read_only=True):
        if is_read_only in [True, 1]:
            file_attributes = os.stat(self.path).st_mode
            os.chmod(self.path, file_attributes & ~stat.S_IWUSR)
        else:
            file_attributes = os.stat(self.path).st_mode
            os.chmod(self.path, file_attributes | stat.S_IWRITE)

