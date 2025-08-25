import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import orjson
import pandas as pd
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder

from langflow.api.v2.files import upload_user_file
from langflow.custom import Component
from langflow.io import DropdownInput, MessageTextInput, StrInput
from langflow.schema import Data, DataFrame, Message
from langflow.services.database.models.user.crud import get_user_by_id
from langflow.services.deps import get_settings_service, get_storage_service, session_scope
from langflow.template.field.base import Output


class SaveToFileComponent(Component):
    display_name = "Save File"
    description = "Save data to a local file in the selected format. Can be used as a tool in agent flows."
    documentation: str = "https://docs.langflow.org/components-processing#save-file"
    icon = "save"
    name = "SaveToFile"

    # File format options for different types
    DATA_FORMAT_CHOICES = ["csv", "excel", "json", "markdown"]
    MESSAGE_FORMAT_CHOICES = ["txt", "json", "markdown"]

    inputs = [
        MessageTextInput(
            name="input",
            display_name="Input",
            info="The input to save (text content).",
            required=True,
            tool_mode=True,  # Enable tool mode for this input
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            info="Name file will be saved as (without extension).",
            required=True,
            tool_mode=True,  # Enable tool mode for this input
        ),
        StrInput(
            name="folder_path",
            display_name="Folder Path",
            info="Path to the folder where the file should be saved. If not provided, saves to current working directory.",
            required=False,
            tool_mode=True,  # Enable tool mode for this input
            advanced=True,
        ),
        DropdownInput(
            name="file_format",
            display_name="File Format",
            options=list(dict.fromkeys(DATA_FORMAT_CHOICES + MESSAGE_FORMAT_CHOICES)),
            info="Select the file format to save the input. If not provided, the default format will be used.",
            value="",
            advanced=True,
            tool_mode=True,  # Enable tool mode for this input
        ),
    ]

    outputs = [
        Output(display_name="File Path", name="message", method="save_to_file", tool_mode=True),
        Output(display_name="Tool", name="tool", method="build_tool", tool_mode=True)
    ]

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.add_tool_output = True  # Enable tool output generation

    async def save_to_file(self) -> Message:
        """Save the input to a file and upload it, returning a confirmation message."""
        # Validate inputs
        if not self.file_name:
            msg = "File name must be provided."
            raise ValueError(msg)
        if not self.input:
            msg = "Input content must be provided."
            raise ValueError(msg)

        # Validate file format based on input type
        file_format = self.file_format or self._get_default_format()
        allowed_formats = self.MESSAGE_FORMAT_CHOICES  # Since we're using MessageTextInput
        if file_format not in allowed_formats:
            msg = f"Invalid file format '{file_format}'. Allowed: {allowed_formats}"
            raise ValueError(msg)

        # Prepare file path with folder
        if self.folder_path:
            # Use the specified folder path
            folder_path = Path(self.folder_path).expanduser().resolve()
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
            file_path = folder_path / self.file_name
        else:
            # Use current working directory
            file_path = Path(self.file_name).expanduser()
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_path = self._adjust_file_path_with_format(file_path, file_format)

        # Save the input as a message
        confirmation = await self._save_message_text(self.input, file_path, file_format)

        # Upload the saved file
        await self._upload_file(file_path)

        # Return the final file path and confirmation message
        final_path = file_path.resolve()

        return Message(text=f"{confirmation} at {final_path}")

    async def build_tool(self):
        """Build the tool representation for agent use."""
        # This method is required for tool mode
        # The actual tool functionality is handled by the save_to_file method
        return await self.save_to_file()

    def _get_default_format(self) -> str:
        """Return the default file format for text input."""
        return "txt"

    def _adjust_file_path_with_format(self, path: Path, fmt: str) -> Path:
        """Adjust the file path to include the correct extension."""
        file_extension = path.suffix.lower().lstrip(".")
        if fmt == "excel":
            return Path(f"{path}.xlsx").expanduser() if file_extension not in ["xlsx", "xls"] else path
        return Path(f"{path}.{fmt}").expanduser() if file_extension != fmt else path

    async def _upload_file(self, file_path: Path) -> None:
        """Upload the saved file using the upload_user_file service."""
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        with file_path.open("rb") as f:
            async with session_scope() as db:
                if not self.user_id:
                    msg = "User ID is required for file saving."
                    raise ValueError(msg)
                current_user = await get_user_by_id(db, self.user_id)

                await upload_user_file(
                    file=UploadFile(filename=file_path.name, file=f, size=file_path.stat().st_size),
                    session=db,
                    current_user=current_user,
                    storage_service=get_storage_service(),
                    settings_service=get_settings_service(),
                )

    async def _save_message_text(self, text_content: str, path: Path, fmt: str) -> str:
        """Save text content to the specified file format."""
        if fmt == "txt":
            path.write_text(text_content, encoding="utf-8")
        elif fmt == "json":
            path.write_text(json.dumps({"content": text_content}, indent=2), encoding="utf-8")
        elif fmt == "markdown":
            path.write_text(f"{text_content}", encoding="utf-8")
        else:
            msg = f"Unsupported format: {fmt}"
            raise ValueError(msg)
        return f"Content saved successfully as '{path}'"
