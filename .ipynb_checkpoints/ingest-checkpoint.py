# ingest.py

import streamlit as st
import openai
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.image import UnstructuredImageLoader
import pytesseract

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load the Notion content located in the folder 'notion_content'
notion_loader  = NotionDirectoryLoader("notion_content")
notion_documents  = notion_loader .load()
#print(documents)
#print(documents[0])


# Split the content into smaller chunks
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\n\n","\n","."],
    chunk_size=1500,
    chunk_overlap=100)
notion_docs = markdown_splitter.split_documents(notion_documents )


print("Revisar Data Unstructured")
pytesseract.pytesseract.tesseract_cmd = r'D:\\Program Files\\Tesseract-OCR\\tesseract.exe'


#image_loader = UnstructuredImageLoader("notion_content/R clase 2 d16f384338f84535b4029c238172b1b2/Untitled 1.png")
#image_documents = image_loader.load()
#print(image_documents)
#print(image_documents[0])
#image_docs = markdown_splitter.split_documents(image_documents)



"""
Google Calendar reader.
"""

import datetime
import os
from typing import Any, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import BaseComponent 
#from llama_index.core.schema import Document
from llama_index.core import Document

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class GoogleCalendarReader(BaseReader):
    """Google Calendar reader.

    Reads events from Google Calendar

    """

    def load_data(
        self,
        number_of_results: Optional[int] = 100,
        start_date: Optional[Union[str, datetime.date]] = None,
    ) -> List[Document]:

        """Load data from user's calendar.

        Args:
            number_of_results (Optional[int]): the number of events to return. Defaults to 100.
            start_date (Optional[Union[str, datetime.date]]): the start date to return events from. Defaults to today.
        """

        from googleapiclient.discovery import build
        print("load_data de google calendar")
        credentials = self._get_credentials()
        service = build("calendar", "v3", credentials=credentials)

        if start_date is None:
            start_date = datetime.date.today()
        elif isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)

        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        start_datetime_utc = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=start_datetime_utc,
                maxResults=number_of_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = events_result.get("items", [])

        if not events:
            return []

        results = []
        for event in events:
            if "dateTime" in event["start"]:
                start_time = event["start"]["dateTime"]
            else:
                start_time = event["start"]["date"]

            if "dateTime" in event["end"]:
                end_time = event["end"]["dateTime"]
            else:
                end_time = event["end"]["date"]

            event_string = f"Status: {event['status']}, "
            event_string += f"Summary: {event['summary']}, "
            event_string += f"Start time: {start_time}, "
            event_string += f"End time: {end_time}, "

            organizer = event.get("organizer", {})
            display_name = organizer.get("displayName", "N/A")
            print(event_string)
            email = organizer.get("email", "N/A")
            if display_name != "N/A":
                event_string += f"Organizer: {display_name} ({email})"
            else:
                event_string += f"Organizer: {email}"
            results.append(Document(text=event_string))
        return results

    def _get_credentials(self) -> Any:
        """Get valid user credentials from storage.

        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.

        Returns:
            Credentials, the obtained credential.
        """
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=3030)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds
'''
The following makes a request to Google Calendar API with the OAuth token
and retrieves number_of_results from the specified date.
'''

#from GoogleCalendarReader import GoogleCalendarReader
from datetime import date

calendar_loader = GoogleCalendarReader()
calendar_documents = calendar_loader.load_data(start_date=date.today(), number_of_results=50)


from typing import List
from langchain.docstore.document import Document as LCDocument

formatted_calendar_documents: List[LCDocument] = [doc.to_langchain_format() for doc in calendar_documents]





from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

'''
OpenAIEmbeddings uses text-embedding-ada-002
'''

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
calendar_docs  = text_splitter.split_documents(formatted_calendar_documents)

#all_documents = notion_docs + image_docs + calendar_docs
all_documents = notion_docs + calendar_docs

 


# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Convert all chunks into vectors embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save to local folder 'faiss_index'
db = FAISS.from_documents(all_documents, embeddings)
db.save_local("faiss_index")

print('Local FAISS index has been successfully saved.')

