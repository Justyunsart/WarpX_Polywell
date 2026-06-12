"""
Storage backend abstraction.

Usage:
    backend = get_backend()
    if not backend.exists("my_file.h5"):
        # ... generate file locally at tmp_path ...
        backend.save(tmp_path, "my_file.h5")
"""
import shutil
from pathlib import Path
from warpx_polywell.utils.config import get_config
import os, pickle

# Google Drive API imports are intentionally deferred into DriveBackend
# methods so users on STORAGE_BACKEND=local don't need the Google packages
# installed at all. Importing this module must remain side-effect-free.


class LocalBackend:
    """Saves files to the local filesystem under LOCAL_OUTPUT_DIR/<subdir>."""

    def __init__(self, subdir: str = ""):
        cfg = get_config()
        self.base_dir = Path(cfg["LOCAL_OUTPUT_DIR"]) / subdir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def exists(self, name: str) -> bool:
        """Returns True if a file with *name* already exists in the output dir."""
        return (self.base_dir / name).is_file()

    def resolve(self, name: str) -> Path:
        """Returns the full local path for *name* (file need not exist yet)."""
        return self.base_dir / name

    def save(self, local_path: Path, name: str) -> Path:
        """
        Ensures *local_path* is stored under the backend's output dir as *name*.
        If *local_path* is already at the correct destination, this is a no-op.
        Returns the final path.
        """
        dest = self.base_dir / name
        local_path = Path(local_path)
        if local_path.resolve() != dest.resolve():
            shutil.move(str(local_path), dest)
        return dest

    def rename(self, old_name: str, new_name: str) -> Path:
        """Renames a file within the backend's output dir. Returns the new path."""
        old_path = self.base_dir / old_name
        new_path = self.base_dir / new_name
        old_path.rename(new_path)
        return new_path


class DriveBackend:
    """
    Saves files to a Google Drive folder.

    Requires the `google-api-python-client` and `google-auth` packages,
    plus a GOOGLE_DRIVE_FOLDER_ID in .env.

    File existence is checked by listing the Drive folder; files are
    downloaded to a local cache for read-back access.
    """

    def __init__(self, subdir: str = ""):
        cfg = get_config()
        self.folder_id = cfg.get("GOOGLE_DRIVE_FOLDER_ID", "")
        if not self.folder_id:
            raise ValueError(
                "GOOGLE_DRIVE_FOLDER_ID is not set in .env. "
                "Set it to use DriveBackend."
            )
        self._subdir = subdir
        self._service = self._build_service()

    def _build_service(self):
        # Lazy imports — only pulled in when DriveBackend is actually used,
        # so local-backend users don't need these packages installed.
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError as e:
            raise ImportError(
                "DriveBackend requires the Google client libraries. Install with:\n"
                "    pip install google-api-python-client "
                "google-auth google-auth-oauthlib"
            ) from e

        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        creds = None
        token_path = os.path.expanduser("~/.config/warpx/token.pickle")
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        creds_path = get_config().get("GOOGLE_DRIVE_CREDENTIALS_PATH")

        if os.path.exists(token_path):
            with open(token_path, 'rb') as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_path, 'wb') as f:
                pickle.dump(creds, f)

        return build('drive', 'v3', credentials=creds)

    def _find_file_id(self, name: str) -> str | None:
        query = (
            f"name='{name}' and '{self.folder_id}' in parents and trashed=false"
        )
        resp = (
            self._service.files()
            .list(q=query, fields="files(id)", pageSize=1)
            .execute()
        )
        files = resp.get("files", [])
        return files[0]["id"] if files else None

    def exists(self, name: str) -> bool:
        return self._find_file_id(name) is not None

    def resolve(self, name: str) -> Path:
        """
        Returns a local temp path where the file should be written before upload.
        The path is under /tmp and need not exist yet.
        """
        import tempfile
        return Path(tempfile.gettempdir()) / name

    def save(self, local_path: Path, name: str) -> Path:
        """
        Uploads *local_path* to Drive as *name* inside the configured folder.
        Returns the local path so callers always have a usable filesystem path.
        """
        from googleapiclient.http import MediaFileUpload

        local_path = Path(local_path)
        media = MediaFileUpload(str(local_path), resumable=True)
        file_id = self._find_file_id(name)
        if file_id:
            self._service.files().update(
                fileId=file_id, media_body=media
            ).execute()
        else:
            metadata = {"name": name, "parents": [self.folder_id]}
            self._service.files().create(
                body=metadata, media_body=media, fields="id"
            ).execute()
        return local_path

    def download(self, name: str) -> Path:
        """
        Downloads *name* from Drive to a local temp file and returns its path.
        Raises FileNotFoundError if the file is not found in Drive.
        """
        import tempfile
        from googleapiclient.http import MediaIoBaseDownload
        import io

        file_id = self._find_file_id(name)
        if file_id is None:
            raise FileNotFoundError(f"'{name}' not found in Drive folder.")
        local_path = Path(tempfile.gettempdir()) / name
        request = self._service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return local_path

    def rename(self, old_name: str, new_name: str) -> str:
        """Renames a file in Drive. Returns the new logical name."""
        file_id = self._find_file_id(old_name)
        if file_id is None:
            raise FileNotFoundError(f"'{old_name}' not found in Drive folder.")
        self._service.files().update(
            fileId=file_id, body={"name": new_name}
        ).execute()
        return new_name


def get_backend(subdir: str = "") -> LocalBackend | DriveBackend:
    """
    Factory that returns the correct backend based on the STORAGE_BACKEND
    config value ('local' or 'drive').
    """
    cfg = get_config()
    backend_type = cfg.get("STORAGE_BACKEND", "local").lower()
    if backend_type == "drive":
        return DriveBackend(subdir=subdir)
    return LocalBackend(subdir=subdir)
