import { HttpRequest, InvocationContext, HttpResponseInit, app } from '@azure/functions';
import { AIChatCompletionRequest } from '@microsoft/ai-chat-protocol';
import { MongoClient } from 'mongodb';
import 'dotenv/config';
import { badRequest, data, serviceUnavailable } from '../http-response';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { createJsonStream } from './chat-post';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';

const currentDate = new Date().toISOString(); // Obtenir la date actuelle en UTC
const systemPrompt = `L'assistant aide les clients de la société de livraison Top Chrono avec leurs questions et demandes de support. Soyez bref dans vos réponses. Répondez uniquement en texte brut, NE PAS utiliser Markdown.
      Répondez UNIQUEMENT avec les informations provenant des sources ci-dessous. S'il n'y a pas assez d'informations dans les sources, dites que vous ne savez pas. Ne générez pas de réponses qui n'utilisent pas les sources. Si poser une question de clarification à l'utilisateur peut aider, posez la question.
      Répondez toujours en français, quelle que soit la langue utilisée dans la question de l'utilisateur.
      La date actuelle est ${currentDate}.
      Générez 3 questions de suivi très brèves que l'utilisateur pourrait poser ensuite.
      Encadrez les questions de suivi entre double chevrons. Exemple :
      <<Puis-je inviter des amis pour une fête?>>
      <<Comment puis-je demander un remboursement?>>
      <<Que se passe-t-il si je casse quelque chose?>>
      Ne répétez pas les questions qui ont déjà été posées.
      Assurez-vous que la dernière question se termine par ">>".
`;

async function searchDocuments(query: string) {
  console.log('Starting searchDocuments function');
  let client: MongoClient;
  const uri = process.env.MONGODB_CONNECTION_STRING || '';
  console.log('MongoDB connection string:', uri);
  const openAiApiKey = process.env.OPENAI_API_KEY;
  try {
    // Initialisation des embeddings OpenAI
    console.log('Initializing OpenAI embeddings');
    const embeddings = new OpenAIEmbeddings({ apiKey: openAiApiKey });
    const queryEmbedding = await embeddings.embedQuery(query);

    // Connexion à MongoDB
    console.log('Connecting to MongoDB');
    client = new MongoClient(uri);
    await client.connect();
    console.log('Connected to MongoDB');

    const database = client.db('serverless');
    console.log('Connected to the database');
    const collection = database.collection('bikerOrder');

    // Définition de l'agrégation pour la recherche vectorielle
    const agg = [
      {
        $vectorSearch: {
          exact: false,
          index: 'new',
          limit: 10,
          numCandidates: 154,
          path: 'information',
          queryVector: queryEmbedding,
          similarity: 'D',
        },
      },
      {
        $project: {
          _id: 0,
          content: 1,
          score: {
            $meta: 'vectorSearchScore',
          },
        },
      },
    ];

    // Exécution de l'agrégation
    const result = await collection.aggregate(agg).toArray();
    console.log('Search result:', result);
    return result;
  } catch (error) {
    console.error('Error in searchDocuments function:', error);
    throw new Error('Failed to search documents');
  }
}

export async function postChatMongo(request: HttpRequest, context: InvocationContext): Promise<HttpResponseInit> {
  try {
    const openAiApiKey = process.env.OPENAI_API_KEY;
    const requestBody = (await request.json()) as AIChatCompletionRequest;

    const { messages } = requestBody;
    if (!messages || messages.length === 0 || !messages.at(-1)?.content) {
      return badRequest('Invalid or missing messages in the request body');
    }

    const lastUserMessage = messages.at(-1)!.content;
    const documents = await searchDocuments(lastUserMessage);
    const contextText = documents.map((doc: any) => doc.content).join('\n');

    const model = new ChatOpenAI({
      apiKey: openAiApiKey,
      model: 'gpt-3.5-turbo',
      temperature: 0.7,
    });

    const prompt = ChatPromptTemplate.fromMessages([
      ['system', systemPrompt],
      [
        'human',
        `Utilisez les informations suivantes pour répondre à la question :\n\n${contextText}\n\nQuestion : ${lastUserMessage}\nRéponse :`,
      ],
    ]);
    const chain = prompt.pipe(model);
    const responseStream = await chain.stream({
      input: lastUserMessage,
    });
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

app.setup({ enableHttpStream: true });
app.http('chat-post-mongo', {
  route: 'chat/mongo/stream',
  methods: ['POST'],
  authLevel: 'anonymous',
  handler: postChatMongo,
});
