/**
 * Cloudflare Worker for RAG Blog System
 * Proxies requests to Gemini API with secure API key
 */

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

/**
 * Handle incoming requests
 */
export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: corsHeaders,
      });
    }

    // Only allow POST requests
    if (request.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Method not allowed' }), {
        status: 405,
        headers: {
          'Content-Type': 'application/json',
          ...corsHeaders,
        },
      });
    }

    try {
      // Parse request body
      const { question, context } = await request.json();

      // Validate input
      if (!question || !context || !Array.isArray(context)) {
        return new Response(
          JSON.stringify({
            error: 'Invalid request. Expected { question: string, context: string[] }',
          }),
          {
            status: 400,
            headers: {
              'Content-Type': 'application/json',
              ...corsHeaders,
            },
          }
        );
      }

      // Check if API key is configured
      if (!env.GEMINI_API_KEY) {
        console.error('GEMINI_API_KEY not configured');
        return new Response(
          JSON.stringify({
            error: 'Service configuration error',
          }),
          {
            status: 500,
            headers: {
              'Content-Type': 'application/json',
              ...corsHeaders,
            },
          }
        );
      }

      // Build the prompt
      const prompt = buildPrompt(question, context);

      // Call Gemini API
      const answer = await callGeminiAPI(prompt, env.GEMINI_API_KEY);

      // Return the answer
      return new Response(
        JSON.stringify({
          answer: answer,
          question: question,
        }),
        {
          status: 200,
          headers: {
            'Content-Type': 'application/json',
            ...corsHeaders,
          },
        }
      );
    } catch (error) {
      console.error('Error processing request:', error);

      return new Response(
        JSON.stringify({
          error: 'Internal server error',
          message: error.message,
        }),
        {
          status: 500,
          headers: {
            'Content-Type': 'application/json',
            ...corsHeaders,
          },
        }
      );
    }
  },
};

/**
 * Build the prompt for Gemini API
 */
function buildPrompt(question, context) {
  const contextText = context.join('\n\n---\n\n');

  return `Tu es un assistant qui répond à des questions sur un article technique de blog.

CONTEXTE DE L'ARTICLE:
${contextText}

QUESTION DU LECTEUR:
${question}

INSTRUCTIONS:
- Réponds uniquement en te basant sur le contexte fourni
- Si la réponse n'est pas dans le contexte, dis-le clairement
- Sois précis et concis (2-3 paragraphes maximum)
- Utilise un ton pédagogique et accessible
- Si pertinent, cite des passages spécifiques du contexte
- Ne mentionne pas que tu as reçu un "contexte" ou des "passages", réponds naturellement

RÉPONSE:`;
}

/**
 * Call Gemini API
 */
async function callGeminiAPI(prompt, apiKey) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      contents: [
        {
          parts: [
            {
              text: prompt,
            },
          ],
        },
      ],
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 1024,
      },
      safetySettings: [
        {
          category: 'HARM_CATEGORY_HARASSMENT',
          threshold: 'BLOCK_MEDIUM_AND_ABOVE',
        },
        {
          category: 'HARM_CATEGORY_HATE_SPEECH',
          threshold: 'BLOCK_MEDIUM_AND_ABOVE',
        },
        {
          category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
          threshold: 'BLOCK_MEDIUM_AND_ABOVE',
        },
        {
          category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
          threshold: 'BLOCK_MEDIUM_AND_ABOVE',
        },
      ],
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('Gemini API error:', errorText);
    throw new Error(`Gemini API error: ${response.status}`);
  }

  const data = await response.json();

  // Extract the text from Gemini response
  if (
    data.candidates &&
    data.candidates[0] &&
    data.candidates[0].content &&
    data.candidates[0].content.parts &&
    data.candidates[0].content.parts[0]
  ) {
    return data.candidates[0].content.parts[0].text;
  } else {
    throw new Error('Invalid response from Gemini API');
  }
}
