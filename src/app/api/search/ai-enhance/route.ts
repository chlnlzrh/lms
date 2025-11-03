import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || '',
})

export async function POST(request: NextRequest) {
  try {
    const { query, results } = await request.json()

    if (!process.env.ANTHROPIC_API_KEY) {
      return NextResponse.json(
        { error: 'AI enhancement unavailable - API key not configured' },
        { status: 503 }
      )
    }

    if (!query) {
      return NextResponse.json(
        { error: 'Query is required' },
        { status: 400 }
      )
    }

    // Create a prompt for AI to analyze the search query and results
    const prompt = `You are an AI learning assistant for an LMS platform. A user searched for: "${query}"

Here are the top search results:
${results.map((result: any, index: number) => `
${index + 1}. ${result.title} (${result.track} - ${result.difficulty})
   Description: ${result.description}
   Type: ${result.type}
`).join('')}

Please provide:
1. A brief explanation of what the user is likely looking for
2. Why these results are relevant
3. Suggested learning path or next steps
4. Any related topics they might also be interested in

Keep your response concise (2-3 sentences) and helpful for learning.`

    const response = await anthropic.messages.create({
      model: 'claude-3-haiku-20240307',
      max_tokens: 300,
      messages: [
        {
          role: 'user',
          content: prompt
        }
      ]
    })

    const insights = response.content[0].type === 'text' ? response.content[0].text : ''

    // Also generate individual explanations for each result
    const enhancedResults = await Promise.all(
      results.slice(0, 3).map(async (result: any) => {
        try {
          const explanationPrompt = `Why would the lesson "${result.title}" (${result.track} track, ${result.difficulty} level) be relevant for someone searching for "${query}"? 

Lesson description: ${result.description}

Provide a one-sentence explanation of the relevance.`

          const explanationResponse = await anthropic.messages.create({
            model: 'claude-3-haiku-20240307',
            max_tokens: 100,
            messages: [
              {
                role: 'user',
                content: explanationPrompt
              }
            ]
          })

          const explanation = explanationResponse.content[0].type === 'text' 
            ? explanationResponse.content[0].text 
            : ''

          return {
            ...result,
            aiExplanation: explanation
          }
        } catch (error) {
          console.error('Error generating explanation for result:', error)
          return result
        }
      })
    )

    return NextResponse.json({
      insights,
      enhancedResults
    })

  } catch (error) {
    console.error('AI enhancement error:', error)
    return NextResponse.json(
      { error: 'AI enhancement temporarily unavailable' },
      { status: 500 }
    )
  }
}