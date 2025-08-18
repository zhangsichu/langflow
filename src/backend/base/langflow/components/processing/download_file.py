import re
import mimetypes
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
    """

    display_name = "Download File"
    description = "Download files from URLs and upload them to Langflow storage."
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
            
            logger.info(f"Successfully uploaded {file_info['filename']} to storage")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file {file_info['filename']} to storage: {e}")
            return False

    async def download_files(self) -> DataFrame:
        """Download files from URLs and return results as DataFrame."""
        urls = self._extract_urls_from_input()
        
        if not urls:
            logger.warning("No URLs found to download")
            return DataFrame(data=[])
        
        logger.info(f"Found {len(urls)} URLs to process")
        
        results = []
        for url in urls:
            result = self._download_file(url)
            if result:
                # Upload to storage if download was successful
                if result["status"] == "success":
                    await self._upload_file_to_storage(result)
                
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
        
        # Count successful downloads
        successful = sum(1 for r in results_df.data if r.get("status") == "success")
        total = len(results_df.data)
        
        summary = f"Downloaded {successful}/{total} files successfully."
        
        # Create detailed message
        details = []
        for result in results_df.data:
            if result.get("status") == "success":
                details.append(f"✅ {result['filename']} ({result['size_mb']} MB)")
            else:
                details.append(f"❌ {result['url']} - {result.get('error', 'Unknown error')}")
        
        message_text = f"{summary}\n\n" + "\n".join(details)
        
        return Message(
            text=message_text,
            data={"data": results_df.data, "summary": summary, "successful": successful, "total": total}
        )
