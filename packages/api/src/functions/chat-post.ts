import { Readable } from 'node:stream';
import { HttpRequest, InvocationContext, HttpResponseInit, app } from '@azure/functions';
import { AIChatCompletionRequest } from '@microsoft/ai-chat-protocol';
import { AzureOpenAIEmbeddings, AzureChatOpenAI } from '@langchain/openai';
import { Embeddings } from '@langchain/core/embeddings';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { VectorStore } from '@langchain/core/vectorstores';
import { OllamaEmbeddings } from '@langchain/community/embeddings/ollama';
import { ChatOllama } from '@langchain/community/chat_models/ollama';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { AzureAISearchVectorStore } from '@langchain/community/vectorstores/azure_aisearch';
import { OpenAIEmbeddings, ChatOpenAI } from '@langchain/openai';
import { badRequest, data, serviceUnavailable } from '../http-response';
import { ollamaChatModel, ollamaEmbeddingsModel } from '../constants';
import { getAzureOpenAiTokenProvider, getCredentials } from '../security';
import { BaseMessageChunk } from '@langchain/core/messages';

const systemPrompt = `L'assistant aide les clients de la société de livraison Top Chrono avec leurs questions et demandes de support. Soyez bref dans vos réponses. Répondez uniquement en texte brut, NE PAS utiliser Markdown.
  Répondez UNIQUEMENT avec les informations provenant des sources ci-dessous. S'il n'y a pas assez d'informations dans les sources, dites que vous ne savez pas. Ne générez pas de réponses qui n'utilisent pas les sources. Si poser une question de clarification à l'utilisateur peut aider, posez la question.
  Si la question de l'utilisateur n'est pas en français, répondez dans la langue utilisée dans la question.

  Chaque source a le format "[nom du fichier] : information". Référencez TOUJOURS le nom du fichier source pour chaque partie utilisée dans la réponse. Utilisez le format "[nom du fichier]" pour référencer une source, par exemple : [info1.txt]. Listez chaque source séparément, par exemple : [info1.txt][info2.pdf].

  Générez 3 questions de suivi très brèves que l'utilisateur pourrait poser ensuite.
  Encadrez les questions de suivi entre double chevrons. Exemple :
  <<Puis-je inviter des amis pour une fête?>>
  <<Comment puis-je demander un remboursement?>>
  <<Que se passe-t-il si je casse quelque chose?>>

  Ne répétez pas les questions qui ont déjà été posées.
  Assurez-vous que la dernière question se termine par ">>".

`;

export async function postChat(request: HttpRequest, context: InvocationContext): Promise<HttpResponseInit> {
  const azureOpenAiEndpoint = process.env.AZURE_OPENAI_API_ENDPOINT;
  const openAiApiKey = process.env.OPEN_AI_API_KEY;
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
    } else if (openAiApiKey) {
      // If OpenAI API key is set, use OpenAI models
      context.log('Using OpenAI models');
      embeddings = new OpenAIEmbeddings({ apiKey: openAiApiKey });
      model = new ChatOpenAI({
        apiKey: openAiApiKey,
        model: 'gpt-3.5-turbo',
        temperature: 0.7,
      });
    } else {
      // If no environment variables are set, it means we are running locally
      context.log('No Azure OpenAI endpoint or OpenAI API key set, using Ollama models and local DB');
      embeddings = new OllamaEmbeddings({ model: ollamaEmbeddingsModel });
      model = new ChatOllama({ model: ollamaChatModel });
    }

    store = await FaissStore.fromTexts(
      [
        'Top Chrono propose une flotte de véhicules allant des vélos aux utilitaires de 20 m3 pour expédier vos colis. La majorité de nos véhicules sont disponibles en version électrique ou GNV pour réduire les émissions. Que ce soit pour un pli ou un colis volumineux, nous avons la solution adaptée à votre besoin, peu importe la zone de livraison !',
        'Top Chrono offre un service de livraison express pour vos colis urgents. Nos coursiers sont disponibles 24h/24 et 7j/7 pour garantir une livraison rapide et fiable.',
        "Notre plateforme en ligne vous permet de commander vos courses en quelques clics. Vous pouvez suivre l'état de votre commande en temps réel et être informé de chaque étape de la livraison.",
        'Votre commande est traitée selon le degré d’urgence souhaité. Notre logiciel assigne automatiquement un coursier pour réaliser la livraison dans les meilleurs délais.',
        'Notre service client est disponible 7j/7 pour résoudre les éventuels imprévus, gérer les reprogrammations de livraison ou répondre à vos questions. Nous sommes là pour vous aider à chaque étape.',
        'Top Chrono propose des solutions de livraison sur mesure pour les entreprises. Que vous ayez besoin de livraisons régulières ou ponctuelles, nous avons des offres adaptées à vos besoins professionnels.',
        'En 2021, Top Chrono a livré plus de 10 millions de colis à travers la France, avec un taux de satisfaction client de 98%.',
        'Top Chrono utilise des technologies avancées pour optimiser les itinéraires de livraison et réduire les délais. Notre système de suivi en temps réel permet aux clients de connaître l’emplacement exact de leur colis à tout moment.',
        'Top Chrono s’engage pour l’environnement en utilisant des véhicules électriques et en compensant les émissions de CO2 de ses livraisons.',
        'Top Chrono a été fondée par Jean Dupin en 1984 à Paris, France. Jean Dupont est un entrepreneur visionnaire qui a vu le potentiel de la livraison express dans une grande ville.',
      ],
      [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }, { id: 6 }, { id: 7 }, { id: 8 }, { id: 9 }, { id: 10 }],
      embeddings,
    );

    const queryEmbedding = await embeddings.embedQuery(messages.at(-1)!.content);
    const results = await store.similaritySearchVectorWithScore(queryEmbedding, 10);
    const contextText = results.map(([doc, score]) => doc.pageContent).join('\n');

    const prompt = ChatPromptTemplate.fromMessages([
      ['system', systemPrompt],
      [
        'human',
        `Utilisez les informations suivantes pour répondre à la question :\n\n${contextText}\n\nQuestion : ${messages.at(-1)!.content}\nRéponse :`,
      ],
    ]);
    const chain = prompt.pipe(model);
    const lastUserMessage = messages.at(-1)!.content;
    const responseStream = await chain.stream({
      input: lastUserMessage,
    });
    // Log the responseStream to verify it contains data
    const jsonStream = createJsonStream(responseStream);

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

export function createJsonStream(chunks: AsyncIterable<BaseMessageChunk>): Readable {
  const buffer = new Readable({
    read() {},
  });

  const stream = async () => {
    for await (const chunk of chunks) {
      if (!chunk.content) continue;

      const responseChunk = {
        index: 0,
        delta: {
          content: chunk.content,
          role: 'assistant',
        },
      };

      // Format response chunks in Newline delimited JSON
      // see https://github.com/ndjson/ndjson-spec
      buffer.push(JSON.stringify(responseChunk) + '\n');
    }

    buffer.push(null);
  };

  stream();

  return buffer;
}

app.setup({ enableHttpStream: true });
app.http('chat-post', {
  route: 'chat/stream',
  methods: ['POST'],
  authLevel: 'anonymous',
  handler: postChat,
});
