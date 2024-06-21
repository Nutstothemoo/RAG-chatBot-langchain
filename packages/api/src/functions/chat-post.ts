import { Readable } from 'node:stream';
import { HttpRequest, InvocationContext, HttpResponseInit, app } from '@azure/functions';
import { AIChatCompletionRequest, AIChatCompletionDelta } from '@microsoft/ai-chat-protocol';
import { Document } from '@langchain/core/documents';
import { AzureOpenAIEmbeddings, AzureChatOpenAI } from '@langchain/openai';
import { Embeddings } from '@langchain/core/embeddings';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { VectorStore } from '@langchain/core/vectorstores';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { ChatOllama } from '@langchain/community/chat_models/ollama';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { AzureAISearchVectorStore } from '@langchain/community/vectorstores/azure_aisearch';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import 'dotenv/config';
import { badRequest, data, serviceUnavailable } from '../http-response';
import { ollamaChatModel, ollamaEmbeddingsModel, faissStoreFolder } from '../constants';
import { getAzureOpenAiTokenProvider, getCredentials } from '../security';

const systemPrompt = `L'assistant aide les clients de notre entreprise Top Chrono reference française de la livraison avec leurs questions et demandes de support. Soyez bref dans vos réponses. Répondez uniquement en texte brut, N'UTILISEZ PAS de Markdown.
Répondez UNIQUEMENT avec les informations provenant des sources ci-dessous. S'il n'y a pas assez d'informations dans les sources, dites que vous ne savez pas. Ne générez pas de réponses qui n'utilisent pas les sources. Si poser une question de clarification à l'utilisateur serait utile, posez la question.
Si la question de l'utilisateur n'est pas en français, répondez dans la langue utilisée dans la question.

Chaque source a le format "[nom du fichier] : information". TOUJOURS référencer le nom du fichier source pour chaque partie utilisée dans la réponse. Utilisez le format "[nom du fichier]" pour référencer une source, par exemple : [info1.txt]. Listez chaque source séparément, par exemple : [info1.txt][info2.pdf].

Générez 3 questions de suivi très brèves que l'utilisateur pourrait poser ensuite.
Encadrez les questions de suivi avec des doubles chevrons. Exemple :
<<Puis-je inviter des amis à une fête?>>
<<Comment puis-je demander un remboursement?>>
<<Que se passe-t-il si je casse quelque chose?>>

Ne répétez pas les questions qui ont déjà été posées.
Assurez-vous que la dernière question se termine par ">>".

SOURCES :
{context}`;

export async function postChat(request: HttpRequest, context: InvocationContext): Promise<HttpResponseInit> {
  const azureOpenAiEndpoint = process.env.AZURE_OPENAI_API_ENDPOINT;

  try {
    const requestBody = (await request.json()) as AIChatCompletionRequest;
    const { messages } = requestBody;

    if (!messages || messages.length === 0 || !messages.at(-1)?.content) {
      return badRequest('Invalid or missing messages in the request body');
    }

    let embeddings: Embeddings;
    let model: BaseChatModel;
    let store: VectorStore;

    if (azureOpenAiEndpoint) {
      const credentials = getCredentials();
      const azureADTokenProvider = getAzureOpenAiTokenProvider();

      // Initialize models and vector database
      embeddings = new AzureOpenAIEmbeddings({ azureADTokenProvider });
      model = new AzureChatOpenAI({
        // Controls randomness. 0 = deterministic, 1 = maximum randomness
        temperature: 0.7,
        azureADTokenProvider,
      });
      store = new AzureAISearchVectorStore(embeddings, { credentials });
    } else {
      // If no environment variables are set, it means we are running locally
      context.log('No Azure OpenAI endpoint set, using Ollama models and local DB');
      embeddings = new OllamaEmbeddings({ model: ollamaEmbeddingsModel });
      model = new ChatOllama({
        temperature: 0.7,
        model: ollamaChatModel,
      });
      store = await FaissStore.load(faissStoreFolder, embeddings);
    }

    // Create the chain that combines the prompt with the documents
    const combineDocsChain = await createStuffDocumentsChain({
      llm: model,
      prompt: ChatPromptTemplate.fromMessages([
        ['system', systemPrompt],
        ['human', '{input}'],
      ]),
      documentPrompt: PromptTemplate.fromTemplate('[{source}]: {page_content}\n'),
    });

    // Create the chain to retrieve the documents from the database
    const chain = await createRetrievalChain({
      retriever: store.asRetriever(3),
      combineDocsChain,
    });

    const lastUserMessage = messages.at(-1)!.content;
    const responseStream = await chain.stream({
      input: lastUserMessage,
    });
    const jsonStream = Readable.from(createJsonStream(responseStream));

    return data(jsonStream, {
      'Content-Type': 'application/x-ndjson',
      'Transfer-Encoding': 'chunked',
    });
  } catch (_error: unknown) {
    const error = _error as Error;
    context.error(`Error when processing chat-post request: ${error.message}`);

    return serviceUnavailable('Service temporarily unavailable. Please try again later.');
  }
}

// Transform the response chunks into a JSON stream
async function* createJsonStream(chunks: AsyncIterable<{ context: Document[]; answer: string }>) {
  for await (const chunk of chunks) {
    if (!chunk.answer) continue;

    const responseChunk: AIChatCompletionDelta = {
      delta: {
        content: chunk.answer,
        role: 'assistant',
      },
    };

    // Format response chunks in Newline delimited JSON
    // see https://github.com/ndjson/ndjson-spec
    yield JSON.stringify(responseChunk) + '\n';
  }
}

app.setup({ enableHttpStream: true });
app.http('chat-post', {
  route: 'chat/stream',
  methods: ['POST'],
  authLevel: 'anonymous',
  handler: postChat,
});
