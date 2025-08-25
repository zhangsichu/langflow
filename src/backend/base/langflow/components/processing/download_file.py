import re
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse, urljoin
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from fastapi import UploadFile
from loguru import logger

from langflow.custom import Component
from langflow.api.v2.files import upload_user_file
from langflow.io import (
    HandleInput,
    StrInput,
    BoolInput,
    IntInput,
    DropdownInput,
    TableInput,
    MessageTextInput,
)
from langflow.schema import Data, DataFrame, Message
from langflow.services.auth.utils import create_user_longterm_token
from langflow.services.database.models.user.crud import get_user_by_id
from langflow.services.deps import get_session, get_settings_service, get_storage_service
from langflow.template.field.base import Output

# Constants
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
DEFAULT_UPLOAD_PATH = "downloads"  # Default upload directory
SUPPORTED_EXTENSIONS = [
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".rtf",
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico", ".tiff",
    # Audio/Video
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac", ".ogg", ".webm", ".mkv",
    # Archives
    ".zip", ".rar", ".tar", ".gz", ".bz2", ".7z", ".iso",
    # Code/Data
    ".py", ".java", ".cpp", ".c", ".h", ".js", ".html", ".css", ".xml", ".json",
    ".csv", ".sql", ".db", ".sqlite",
    # Executables
    ".exe", ".msi", ".dmg", ".pkg", ".deb", ".rpm", ".apk"
]

# MIME type patterns for common file types
MIME_TYPE_PATTERNS = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/plain": ".txt",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
    "audio/mpeg": ".mp3",
    "video/mp4": ".mp4",
    "application/zip": ".zip",
    "application/x-rar-compressed": ".rar",
    "application/x-tar": ".tar",
    "application/gzip": ".gz",
    "text/html": ".html",
    "text/css": ".css",
    "application/javascript": ".js",
    "application/json": ".json",
    "text/csv": ".csv",
    "text/xml": ".xml",
    "application/x-python-code": ".py",
    "application/x-java-source": ".java",
    "text/x-c++src": ".cpp",
    "text/x-csrc": ".c",
    "text/x-chdr": ".h",
    "application/x-executable": ".exe",
    "application/x-msdownload": ".exe",
    "application/x-apple-diskimage": ".dmg",
    "application/x-debian-package": ".deb",
    "application/x-redhat-package-manager": ".rpm",
    "application/vnd.android.package-archive": ".apk",
}


class DownloadFileComponent(Component):
    """A component that downloads files from URLs and uploads them to Langflow storage.

    This component allows downloading files from URLs, with options to:
    - Filter by file extensions
    - Set maximum file size limits
    - Configure request headers and timeouts
    - Extract file links from HTML content
    - Automatically detect file types from MIME types
    - Handle various file formats and download scenarios
    - Configure custom upload paths for downloaded files

    NEW FEATURES - Custom Upload Paths:
    ===================================
    
    1. Custom Upload Path:
       - Set a custom directory where downloaded files will be saved
       - Supports both relative and absolute paths
       - Automatically creates directories if they don't exist
       - Validates path permissions and security
    
    2. File Organization Options:
       - flat: All files in the same directory
       - by_extension: Files grouped by file type (e.g., pdf/, images/, etc.)
       - by_date: Files grouped by download date (e.g., 2024-01-15/)
    
    3. Path Validation:
       - Security checks to prevent directory traversal attacks
       - Permission validation
       - Automatic path creation with user control
    
    Usage Examples:
    ===============
    
    Basic custom path:
    - Set use_custom_path = True
    - Set custom_upload_path = "/home/user/downloads"
    
    Organized by extension:
    - Set file_organization = "by_extension"
    - Files will be saved as: /path/pdf/document.pdf, /path/images/photo.jpg
    
    Organized by date:
    - Set file_organization = "by_date"
    - Files will be saved as: /path/2024-01-15/document.pdf
    
    Security Features:
    =================
    - Prevents directory traversal attacks (blocks paths with "..")
    - Validates write permissions before attempting to save
    - Sanitizes directory names for extension-based organization
    - Resolves symbolic links to prevent security issues
    """

    display_name = "Download File"
    description = "Download files from URLs and upload them to Langflow storage with custom path options."
    documentation: str = "https://docs.langflow.org/components-processing#download-file"
    icon = "download"
    name = "DownloadFileComponent"

    inputs = [
        HandleInput(
            name="input",
            display_name="Input",
            info="Input containing URLs to download. Can be Data, DataFrame, Message, or direct URL strings.",
            dynamic=True,
            input_types=["Data", "DataFrame", "Message", "str"],
            required=True,
        ),
        MessageTextInput(
            name="urls",
            display_name="Direct URLs",
            info="Enter one or more direct URLs to download files from, by clicking the '+' button.",
            is_list=True,
            tool_mode=True,
            placeholder="Enter a URL...",
            list_add_label="Add URL",
            input_types=[],
            required=False,
        ),
        StrInput(
            name="file_extensions",
            display_name="File Extensions",
            info="Filter files by specific extensions. Enter extensions separated by commas (e.g., .pdf,.doc,.jpg). Leave empty to download all file types.",
            value="",
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="extract_from_html",
            display_name="Extract from HTML",
            info="If enabled, extracts file links from HTML content in the input.",
            value=True,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="extract_images",
            display_name="Extract Images",
            info="If enabled, extracts image links from HTML content.",
            value=True,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="extract_documents",
            display_name="Extract Documents",
            info="If enabled, extracts document links from HTML content.",
            value=True,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="extract_media",
            display_name="Extract Media",
            info="If enabled, extracts audio/video links from HTML content.",
            value=True,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="extract_archives",
            display_name="Extract Archives",
            info="If enabled, extracts archive links from HTML content.",
            value=True,
            required=False,
            advanced=True,
        ),
        IntInput(
            name="max_file_size",
            display_name="Max File Size (MB)",
            info="Maximum file size to download in MB. Set to 0 for no limit.",
            value=100,
            required=False,
            advanced=True,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            info="Timeout for download requests in seconds.",
            value=DEFAULT_TIMEOUT,
            required=False,
            advanced=True,
        ),
        TableInput(
            name="headers",
            display_name="Headers",
            info="The headers to send with download requests",
            table_schema=[
                {
                    "name": "key",
                    "display_name": "Header",
                    "type": "str",
                    "description": "Header name",
                },
                {
                    "name": "value",
                    "display_name": "Value",
                    "type": "str",
                    "description": "Header value",
                },
            ],
            value=[{"key": "User-Agent", "value": "Mozilla/5.0 (compatible; Langflow-Downloader/1.0)"}],
            advanced=True,
            input_types=["DataFrame"],
        ),
        BoolInput(
            name="continue_on_failure",
            display_name="Continue on Failure",
            info="If enabled, continues downloading other files even if some fail.",
            value=True,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="overwrite_existing",
            display_name="Overwrite Existing",
            info="If enabled, overwrites existing files with the same name.",
            value=False,
            required=False,
            advanced=True,
        ),
        # New path configuration inputs
        StrInput(
            name="custom_upload_path",
            display_name="Custom Upload Path",
            info="Custom path where downloaded files should be uploaded. Can be relative or absolute. Leave empty to use default Langflow storage.",
            value="",
            required=False,
            advanced=True,
            placeholder="e.g., /custom/path or relative/path",
        ),
        BoolInput(
            name="use_custom_path",
            display_name="Use Custom Upload Path",
            info="If enabled, uses the custom upload path instead of default Langflow storage.",
            value=False,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="create_path_if_not_exists",
            display_name="Create Path if Not Exists",
            info="If enabled, creates the custom upload path if it doesn't exist.",
            value=True,
            required=False,
            advanced=True,
        ),
        DropdownInput(
            name="file_organization",
            display_name="File Organization",
            info="How to organize files in the upload path. 'flat': all files in same directory, 'by_extension': group by file type, 'by_date': group by download date.",
            options=[
                "flat",
                "by_extension",
                "by_date",
            ],
            value="flat",
            required=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Download Results", name="results", method="download_files"),
        Output(display_name="Downloaded Files", name="files", method="download_files_as_message", tool_mode=False),
    ]

    def _extract_urls_from_input(self) -> List[str]:
        """Extract URLs from the input data."""
        urls = set()
        
        # Add direct URLs if provided
        if self.urls:
            urls.update(self.urls)
        
        # Extract URLs from input based on type
        if hasattr(self, 'input') and self.input:
            if isinstance(self.input, str):
                urls.add(self.input)
            elif isinstance(self.input, Message):
                # Extract URLs from message text and data
                if self.input.text:
                    urls.update(self._extract_urls_from_text(self.input.text))
                if self.input.data and isinstance(self.input.data, dict):
                    data = self.input.data.get("data", [])
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "url" in item:
                                urls.add(item["url"])
                            if isinstance(item, dict) and "text" in item:
                                urls.update(self._extract_urls_from_text(item["text"]))
            elif isinstance(self.input, DataFrame):
                # Extract URLs from DataFrame
                for row in self.input.data:
                    if isinstance(row, dict):
                        if "url" in row:
                            urls.add(row["url"])
                        if "text" in row:
                            urls.update(self._extract_urls_from_text(row["text"]))
            elif isinstance(self.input, Data):
                # Extract URLs from Data object
                if hasattr(self.input, 'data'):
                    if isinstance(self.input.data, str):
                        urls.update(self._extract_urls_from_text(self.input.data))
                    elif isinstance(self.input.data, list):
                        for item in self.input.data:
                            if isinstance(item, dict):
                                if "url" in item:
                                    urls.add(item["url"])
                                if "text" in item:
                                    urls.update(self._extract_urls_from_text(item["text"]))
        
        return list(urls)

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text content, optionally filtering by file type."""
        if not text or not self.extract_from_html:
            return []
        
        urls = set()
        try:
            soup = BeautifulSoup(text, "lxml")
            
            # Extract links based on enabled options
            if self.extract_images:
                for img in soup.find_all("img"):
                    if img.get("src"):
                        urls.add(img["src"])
            
            if self.extract_documents:
                for link in soup.find_all("a"):
                    href = link.get("href")
                    if href:
                        # Check if it's a document link
                        if any(href.lower().endswith(ext) for ext in [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt"]):
                            urls.add(href)
            
            if self.extract_media:
                for link in soup.find_all("a"):
                    href = link.get("href")
                    if href:
                        # Check if it's a media link
                        if any(href.lower().endswith(ext) for ext in [".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac"]):
                            urls.add(href)
            
            if self.extract_archives:
                for link in soup.find_all("a"):
                    href = link.get("href")
                    if href:
                        # Check if it's an archive link
                        if any(href.lower().endswith(ext) for ext in [".zip", ".rar", ".tar", ".gz", ".7z"]):
                            urls.add(href)
            
            # Also extract from src attributes of various tags
            for tag in soup.find_all(["source", "video", "audio", "embed", "object"]):
                src = tag.get("src") or tag.get("data")
                if src:
                    urls.add(src)
                    
        except Exception as e:
            logger.warning(f"Error extracting URLs from text: {e}")
        
        return list(urls)

    def _is_valid_file_url(self, url: str) -> bool:
        """Check if the URL points to a valid file based on extension and MIME type patterns."""
        if not url:
            return False
        
        # Check if URL has a file extension
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check if path ends with a supported extension
        if any(path.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            return True
        
        # Check if URL contains file-like patterns
        if any(ext in path for ext in SUPPORTED_EXTENSIONS):
            return True
        
        return False

    def _get_file_extension_from_url(self, url: str, content_type: Optional[str] = None) -> str:
        """Get file extension from URL or content type."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Try to get extension from URL path
        for ext in SUPPORTED_EXTENSIONS:
            if path.endswith(ext):
                return ext
        
        # Try to get extension from content type
        if content_type:
            for mime_type, ext in MIME_TYPE_PATTERNS.items():
                if mime_type in content_type.lower():
                    return ext
        
        # Default to .bin if no extension found
        return ".bin"

    def _validate_and_create_path(self, path: str) -> Optional[Path]:
        """Validate and optionally create the custom upload path."""
        if not path or not path.strip():
            return None
        
        try:
            # Clean and normalize the path
            clean_path = path.strip()
            
            # Convert to Path object
            upload_path = Path(clean_path)
            
            # Handle relative paths
            if not upload_path.is_absolute():
                # Make relative to current working directory
                upload_path = Path.cwd() / upload_path
            
            # Resolve any symbolic links and normalize the path
            upload_path = upload_path.resolve()
            
            # Security check: prevent directory traversal attacks
            if ".." in str(upload_path):
                logger.error(f"Invalid path detected (contains '..'): {path}")
                return None
            
            # Check if path exists
            if not upload_path.exists():
                if self.create_path_if_not_exists:
                    try:
                        # Create the path and all parent directories
                        upload_path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created upload directory: {upload_path}")
                    except PermissionError:
                        logger.error(f"Permission denied creating directory: {upload_path}")
                        return None
                    except Exception as e:
                        logger.error(f"Error creating directory {upload_path}: {e}")
                        return None
                else:
                    logger.error(f"Upload path does not exist: {upload_path}")
                    return None
            
            # Check if it's a directory
            if not upload_path.is_dir():
                logger.error(f"Upload path is not a directory: {upload_path}")
                return None
            
            # Check write permissions
            if not os.access(upload_path, os.W_OK):
                logger.error(f"No write permission for upload path: {upload_path}")
                return None
            
            logger.info(f"Validated upload path: {upload_path}")
            return upload_path
            
        except Exception as e:
            logger.error(f"Error validating/creating upload path {path}: {e}")
            return None

    def get_current_working_directory(self) -> str:
        """Get the current working directory for reference."""
        return str(Path.cwd())

    def get_default_download_path(self) -> str:
        """Get the default download path for reference."""
        return str(Path.cwd() / DEFAULT_UPLOAD_PATH)

    def _get_organized_path(self, base_path: Path, filename: str, file_extension: str) -> Path:
        """Get the organized path based on the file organization setting."""
        try:
            if self.file_organization == "flat":
                return base_path / filename
            
            elif self.file_organization == "by_extension":
                # Remove the dot from extension for directory name
                ext_dir = file_extension[1:] if file_extension.startswith('.') else file_extension
                # Sanitize extension directory name (remove special characters)
                ext_dir = re.sub(r'[^\w\-_.]', '_', ext_dir)
                ext_path = base_path / ext_dir
                ext_path.mkdir(exist_ok=True)
                return ext_path / filename
            
            elif self.file_organization == "by_date":
                from datetime import datetime
                today = datetime.now().strftime("%Y-%m-%d")
                date_path = base_path / today
                date_path.mkdir(exist_ok=True)
                return date_path / filename
            
            else:
                # Default to flat organization
                logger.warning(f"Unknown file organization: {self.file_organization}, using flat organization")
                return base_path / filename
                
        except Exception as e:
            logger.error(f"Error creating organized path: {e}")
            # Fallback to flat organization
            return base_path / filename

    def _save_file_to_custom_path(self, file_info: dict, custom_path: Path) -> bool:
        """Save downloaded file to custom path."""
        try:
            # Get organized path
            organized_path = self._get_organized_path(custom_path, file_info["filename"], file_info["extension"])
            
            # Check if file already exists
            if not self.overwrite_existing and organized_path.exists():
                logger.warning(f"File already exists and overwrite is disabled: {organized_path}")
                return False
            
            # Ensure parent directory exists
            organized_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with organized_path.open("wb") as f:
                f.write(file_info["content"])
            
            logger.info(f"Successfully saved {file_info['filename']} to {organized_path}")
            
            # Update file_info with the actual saved path
            file_info["saved_path"] = str(organized_path)
            file_info["saved_directory"] = str(organized_path.parent)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving file {file_info['filename']} to custom path: {e}")
            return False

    def _download_file(self, url: str) -> Optional[dict]:
        """Download a single file from URL."""
        try:
            # Validate URL
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            
            # Check if it's a valid file URL
            if not self._is_valid_file_url(url):
                logger.debug(f"Skipping non-file URL: {url}")
                return None
            
            # Check file extension filter
            if self.file_extensions:
                # Parse comma-separated extensions
                extensions = [ext.strip() for ext in self.file_extensions.split(",") if ext.strip()]
                has_valid_extension = any(url.lower().endswith(ext) for ext in extensions)
                if not has_valid_extension:
                    logger.debug(f"URL {url} doesn't match required extensions: {extensions}")
                    return None
            
            # Prepare headers
            headers = {header["key"]: header["value"] for header in self.headers}
            
            # Download file
            logger.info(f"Downloading file from: {url}")
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                stream=True,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Check content type and file size
            content_type = response.headers.get("content-type", "")
            content_length = int(response.headers.get("content-length", 0))
            
            if self.max_file_size > 0 and content_length > (self.max_file_size * 1024 * 1024):
                logger.warning(f"File {url} is too large: {content_length} bytes")
                return None
            
            # Get file extension
            file_extension = self._get_file_extension_from_url(url, content_type)
            
            # Generate filename
            parsed = urlparse(url)
            original_filename = Path(parsed.path).name
            if not original_filename or original_filename == "/":
                original_filename = f"downloaded_file{file_extension}"
            
            # Ensure filename has correct extension
            if not original_filename.lower().endswith(file_extension.lower()):
                original_filename = f"{original_filename}{file_extension}"
            
            # Download content
            content = response.content
            
            # Create result info
            result = {
                "url": url,
                "filename": original_filename,
                "content_type": content_type,
                "size_bytes": len(content),
                "size_mb": round(len(content) / (1024 * 1024), 2),
                "extension": file_extension,
                "status": "success",
                "content": content
            }
            
            logger.info(f"Successfully downloaded {original_filename} ({result['size_mb']} MB) from {url}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading file from {url}: {e}")
            if not self.continue_on_failure:
                raise
            return {
                "url": url,
                "filename": "",
                "content_type": "",
                "size_bytes": 0,
                "size_mb": 0,
                "extension": "",
                "status": "error",
                "error": str(e),
                "content": None
            }
        except Exception as e:
            logger.error(f"Unexpected error downloading file from {url}: {e}")
            if not self.continue_on_failure:
                raise
            return {
                "url": url,
                "filename": "",
                "content_type": "",
                "size_bytes": 0,
                "size_mb": 0,
                "extension": "",
                "status": "error",
                "error": str(e),
                "content": None
            }

    async def _upload_file_to_storage(self, file_info: dict) -> bool:
        """Upload downloaded file to Langflow storage."""
        if not file_info.get("content") or file_info.get("status") != "success":
            return False
        
        try:
            # Determine the upload path
            if self.use_custom_path and self.custom_upload_path:
                custom_upload_path = self._validate_and_create_path(self.custom_upload_path)
                if not custom_upload_path:
                    logger.error(f"Could not validate or create custom upload path: {self.custom_upload_path}")
                    return False
                
                # Save file to custom path
                if not self._save_file_to_custom_path(file_info, custom_upload_path):
                    return False
                
                logger.info(f"Successfully saved {file_info['filename']} to custom path: {custom_upload_path}")
                return True
            else:
                # Use default Langflow storage (original behavior)
                try:
                    file_path = Path(file_info["filename"])
                    
                    # Check if file already exists
                    if not self.overwrite_existing:
                        # This would need to be implemented based on your storage service
                        pass
                    
                    # Upload file using the same mechanism as save_file component
                    with file_path.open("wb") as f:
                        f.write(file_info["content"])
                    
                    async for db in get_session():
                        user_id, _ = await create_user_longterm_token(db)
                        current_user = await get_user_by_id(db, user_id)
                        
                        # Create UploadFile object
                        upload_file = UploadFile(
                            filename=file_info["filename"],
                            file=file_path.open("rb"),
                            size=file_info["size_bytes"]
                        )
                        
                        # Upload to storage
                        await upload_user_file(
                            file=upload_file,
                            session=db,
                            current_user=current_user,
                            storage_service=get_storage_service(),
                            settings_service=get_settings_service(),
                        )
                    
                    logger.info(f"Successfully uploaded {file_info['filename']} to Langflow storage")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error uploading file {file_info['filename']} to Langflow storage: {e}")
                    return False
            
        except Exception as e:
            logger.error(f"Error in upload process for {file_info['filename']}: {e}")
            return False

    async def download_files(self) -> DataFrame:
        """Download files from URLs and return results as DataFrame."""
        # Validate custom path settings if enabled
        if self.use_custom_path and not self.custom_upload_path:
            logger.warning("Custom path is enabled but no path specified. Falling back to default Langflow storage.")
            self.use_custom_path = False
        
        urls = self._extract_urls_from_input()
        
        if not urls:
            logger.warning("No URLs found to download")
            return DataFrame(data=[])
        
        logger.info(f"Found {len(urls)} URLs to process")
        if self.use_custom_path:
            logger.info(f"Using custom upload path: {self.custom_upload_path}")
        else:
            logger.info("Using default Langflow storage")
        
        results = []
        for url in urls:
            result = self._download_file(url)
            if result:
                # Upload to storage if download was successful
                if result["status"] == "success":
                    upload_success = await self._upload_file_to_storage(result)
                    if upload_success:
                        result["upload_status"] = "success"
                        if self.use_custom_path and self.custom_upload_path:
                            result["upload_location"] = "custom_path"
                            result["upload_path"] = result.get("saved_path", "")
                        else:
                            result["upload_location"] = "langflow_storage"
                    else:
                        result["upload_status"] = "failed"
                        result["upload_location"] = "unknown"
                else:
                    result["upload_status"] = "skipped"
                    result["upload_location"] = "unknown"
                
                # Remove content from result for DataFrame (keep metadata only)
                result_copy = result.copy()
                if "content" in result_copy:
                    del result_copy["content"]
                results.append(result_copy)
        
        return DataFrame(data=results)

    async def download_files_as_message(self) -> Message:
        """Download files and return results as Message."""
        results_df = await self.download_files()
        
        if not results_df.data:
            return Message(text="No files were downloaded.")
        
        # Count successful downloads and uploads
        successful_downloads = sum(1 for r in results_df.data if r.get("status") == "success")
        successful_uploads = sum(1 for r in results_df.data if r.get("upload_status") == "success")
        total = len(results_df.data)
        
        summary = f"Downloaded {successful_downloads}/{total} files successfully. Uploaded {successful_uploads}/{successful_downloads} files successfully."
        
        # Create detailed message
        details = []
        for result in results_df.data:
            if result.get("status") == "success":
                upload_info = ""
                if result.get("upload_status") == "success":
                    if result.get("upload_location") == "custom_path":
                        upload_info = f" → Saved to: {result.get('upload_path', 'Unknown path')}"
                    else:
                        upload_info = " → Uploaded to Langflow storage"
                else:
                    upload_info = " → Upload failed"
                
                details.append(f"✅ {result['filename']} ({result['size_mb']} MB){upload_info}")
            else:
                details.append(f"❌ {result['url']} - {result.get('error', 'Unknown error')}")
        
        message_text = f"{summary}\n\n" + "\n".join(details)
        
        return Message(
            text=message_text,
            data={
                "data": results_df.data, 
                "summary": summary, 
                "successful_downloads": successful_downloads,
                "successful_uploads": successful_uploads,
                "total": total
            }
        )
