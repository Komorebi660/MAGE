from search_env import BrowseCompPlusEnv
from agentcore.actions import BaseAction


class SearchAction(BaseAction):
    def __init__(self, env: BrowseCompPlusEnv) -> None:
        self.env = env
        super().__init__(
            action_name="search",
            action_desc="search relevant documents, return ids and snippets of relevant documents.",
            params_doc={"query": "the search query string, a raw text"},
        )

    def __call__(self, query: str) -> str:
        """
        Search the full corpus.

        return: json string
        json format: list of documents
        [
            {
                "docid": str, # the document id
                "snippet": str, # the most relevant snippet of the document
            }
        ]
        """
        if isinstance(query, dict) and "query" in query:
            query = query["query"]
        if not isinstance(query, str) or not query.strip():
            return "Wrong action parameters: query must be a non-empty string."
        return self.env.search(query)


class GetDocumentAction(BaseAction):
    def __init__(self, env: BrowseCompPlusEnv) -> None:
        self.env = env
        super().__init__(
            action_name="get_document",
            action_desc="get the full document by docid.",
            params_doc={"docid": "the docid to retrieve"},
        )

    def __call__(self, docid: str) -> str:
        if isinstance(docid, dict) and "docid" in docid:
            docid = docid["docid"]
        if isinstance(docid, int):
            docid = str(docid)
        if not isinstance(docid, str) or not docid.strip():
            return "Wrong action parameters: docid must be a non-empty string."
        return self.env.get_document(docid)