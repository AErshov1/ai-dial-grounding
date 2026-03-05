import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)

_HOBBY_SEARCH_SYSTEM_PROMPT = """
You are a user search assistant. Your task is to find users from the provided list that match the search criteria by user hobbies.

## INSTRUCTIONS:
1. Analyze the search request to understand what hobbies are being searched for
2. Examine each user in the USER DATA context and determine if they match the search criteria based on their hobbies
3. For matching users, extract and their id and group them by hobbies
4. Be inclusive - if a user partially matches or could potentially match, include them
5. Output the results in JSON format with hobbies as keys and lists of user IDs as values according to the OUTPUT FORMAT

## OUTPUT FORMAT:
{format_instructions}
"""

_HOBBY_SEARH_USER_PROMPT = """
## USER DATA:
{context}

## SEARCH REQUEST:
{query}
"""

_azure_chat = AzureChatOpenAI(
    api_key=SecretStr(API_KEY),
    api_version="2024-05-01-preview",
    azure_endpoint=DIAL_URL,
    azure_deployment="gpt-4o",
)
_azure_embeddings = AzureOpenAIEmbeddings(
    api_key=SecretStr(API_KEY),
    api_version="2024-05-01-preview",
    azure_endpoint=DIAL_URL,
    azure_deployment="text-embedding-3-small-1",
    dimensions=384,
)

class HobbySearchModel(BaseModel):
    hobby: str = Field(description="The hobby that matches the search criteria")
    user_ids: list[str] = Field(description="List of user IDs that match the hobby")

class HobbySearchResponse(BaseModel):
    results: list[HobbySearchModel] = Field(description="List of hobbies with corresponding user IDs")

class HobbySearchAgent:

  def __init__(self):
     self._user_client = UserClient()
     self._vectorstore = None
     self._user_ids_store : list[int] = []

  async def __aenter__(self):
    users = self._retrive_users()
    await self._create_vectorstore(users)
    return self

  async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
    pass


  def _retrive_users(self, batch_size: int = 100) -> list[list[Document]]:
    print("=> Retrieving users...", end="")
    users = self._user_client.get_all_users()
    documents = []
    for user in users:
        documents.append(Document(
            id=user['id'],
            page_content=user['about_me']
        ))

    self._user_ids_store = [doc.id for doc in documents]
    # print("Done! Total users: ", len(documents))
    return [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

  async def _create_vectorstore(self, docs_batches: list[list[Document]]) -> None:
    print(f"=> Creating vectorstore ...", end="")
    vectorstore = Chroma(
        embedding_function=_azure_embeddings,
        collection_name="users"
    )
    vs_tasks = [vectorstore.aadd_documents(documents=docs) for docs in docs_batches]
    await asyncio.gather(*vs_tasks)
    print("Done!")
    self._vectorstore = vectorstore

  async def _update_vectorstore(self, docs_batches: list[list[Document]]):
    print(f"=> Updating vectorstore, user ids in store {len(self._user_ids_store)}...")
    user_tasks = []
    user_ids = set()
    for docs in docs_batches:
      user_to_add = [new_doc.id for new_doc in docs if new_doc.id not in self._user_ids_store]
      user_ids.update([doc.id for doc in docs])
      print(f"=> Users to add: {user_to_add}")
      self._user_ids_store.extend(user_to_add)

      if user_to_add:
          print(f"=> Adding {len(user_to_add)} new users to vectorstore...")
          user_tasks.append(self._vectorstore.aadd_documents(documents=user_to_add))

    user_to_delete = [user_id for user_id in self._user_ids_store if user_id not in user_ids]
    print(f"=> Users to delete: {user_to_delete}")
    if user_to_delete:
      print(f"=> Deleting {len(user_to_delete)} users from vectorstore...")
      user_tasks.append(self._vectorstore.adelete(ids=user_to_delete))

    if user_tasks:
      await asyncio.gather(*user_tasks)
    print("Done! Actions performed:", len(user_tasks))

  def _join_user_info_from_documents(self, documents: list[tuple[Document, float]]) -> str:
    result = []
    for doc_tuple in documents:
        d, _s = doc_tuple
        result.append(f"User:\n  id:  {d.id}\n  about_me: {d.page_content}\n")
    return "\n".join(result)

  async def _search_users(self, query: str, k: int = 5, score_threshold: float = 0.25) -> str:
    docs = self._retrive_users()
    if not self._vectorstore:
      await self._create_vectorstore(docs)
    else:
      await self._update_vectorstore(docs)

    print(f"=> User Searching...", end="")
    vectorstore_users = self._vectorstore.similarity_search_with_relevance_scores(query=query, k=k, score_threshold=score_threshold)
    context = self._join_user_info_from_documents(vectorstore_users)
    print(f"Done! Found {len(vectorstore_users)} users.")
    return context

  async def search(self, query: str) -> HobbySearchResponse:
    context = await self._search_users(query=query)
    llm_parser = PydanticOutputParser(pydantic_object=HobbySearchResponse)
    messages = [
        SystemMessagePromptTemplate.from_template(template=_HOBBY_SEARCH_SYSTEM_PROMPT),
        HumanMessage(_HOBBY_SEARH_USER_PROMPT.format(context=context, query=query))
    ]

    prompt = ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=llm_parser.get_format_instructions())
    print(f"{'-'*36} Prompt {'-'*36}\n{prompt.format_messages()}\n{'-'*80}")

    response : HobbySearchResponse = ( prompt | _azure_chat | llm_parser ).invoke({})
    get_user_tasks = []
    for hobby in response.results:
        print(f"=> Grounding for hobby '{hobby.hobby}' with {len(hobby.user_ids)} users...")
        full_user_info_tasks = [self._user_client.get_user(id=user_id) for user_id in hobby.user_ids]
        get_user_tasks.append(asyncio.gather(*full_user_info_tasks))

    user_info_results = await asyncio.gather(*get_user_tasks)
    output = []
    for users in user_info_results:
        for u in users:
          str_user = "User:\n"
          for key, value in u.items():
            str_user += f"  {key}: {value}\n"
          output.append(str_user)

    return "\n".join(output)


async def main():
    print("Query samples:")
    print(" - Find people who love mountain climbing and hiking")

    async with HobbySearchAgent() as agent:
      while True:
        user_input = input("> ").strip()
        if user_input:
            if user_input in ["/exit", "/quit"]:
                print("Goodbye!")
                return

            response = await agent.search(user_input)
            print(f"{'='*30} RESPONSE {'='*30}\n{response}\n\n{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
