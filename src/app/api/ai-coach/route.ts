import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface AICoachRequest {
  message: string
  lessonContext: {
    title: string
    content: string
    track: string
    moduleNumber?: number
    estimatedReadTime?: number
    complexity?: string
    topics?: string[]
    learningObjectives?: string[]
  }
  conversationHistory: ChatMessage[]
}

export async function POST(request: NextRequest) {
  try {
    const body: AICoachRequest = await request.json()
    const { message, lessonContext, conversationHistory } = body

    if (!message || !lessonContext) {
      return NextResponse.json(
        { error: 'Message and lesson context are required' },
        { status: 400 }
      )
    }

    // Create system prompt with lesson context
    const systemPrompt = `You are an AI Learning Coach for the LMS platform, specifically helping students with data engineering and AI training lessons. 

CURRENT LESSON CONTEXT:
- Title: ${lessonContext.title}
- Track: ${lessonContext.track === 'data-engineering' ? 'Data Engineering' : 'AI Training'}
- Module: ${lessonContext.moduleNumber ? `Module ${lessonContext.moduleNumber}` : 'N/A'}
- Complexity: ${lessonContext.complexity || 'N/A'}
- Estimated Read Time: ${lessonContext.estimatedReadTime ? `${lessonContext.estimatedReadTime} minutes` : 'N/A'}
- Topics: ${lessonContext.topics?.join(', ') || 'N/A'}
- Learning Objectives: ${lessonContext.learningObjectives?.join('; ') || 'N/A'}

LESSON CONTENT SUMMARY:
${lessonContext.content.substring(0, 1500)}${lessonContext.content.length > 1500 ? '...' : ''}

YOUR ROLE:
- Act as a knowledgeable, supportive learning coach
- Help students understand the current lesson content
- Answer questions related to the lesson material
- Provide clarifications, examples, and additional insights
- Encourage learning and provide motivation
- Use a friendly, professional, and encouraging tone
- Keep responses concise but comprehensive (2-4 sentences typically)
- If asked about topics outside the lesson context, gently redirect to the current lesson

GUIDELINES:
- Always relate answers back to the current lesson when possible
- Provide practical examples from data engineering/AI when relevant
- Encourage hands-on practice and application
- If a student seems confused, break down concepts into simpler terms
- Celebrate student progress and understanding
- Use emojis sparingly but appropriately to maintain engagement`

    // Prepare conversation messages for Claude
    const messages: Array<{ role: 'user' | 'assistant', content: string }> = [
      ...conversationHistory,
      { role: 'user', content: message }
    ]

    // Call Claude Haiku 4.5
    const response = await anthropic.messages.create({
      model: 'claude-3-5-haiku-20241022', // Using the latest available Haiku model
      max_tokens: 500,
      temperature: 0.7,
      system: systemPrompt,
      messages: messages
    })

    const aiResponse = response.content[0]?.type === 'text' 
      ? response.content[0].text 
      : 'I apologize, but I encountered an issue generating a response. Please try again.'

    return NextResponse.json({
      message: aiResponse,
      usage: {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens
      }
    })

  } catch (error) {
    console.error('AI Coach API error:', error)
    
    // Check if it's an API key issue
    if (error instanceof Error && error.message.includes('api_key')) {
      return NextResponse.json(
        { error: 'AI Coach service is currently unavailable. Please check API configuration.' },
        { status: 503 }
      )
    }

    return NextResponse.json(
      { error: 'Failed to get AI response. Please try again.' },
      { status: 500 }
    )
  }
}

// Health check endpoint
export async function GET() {
  return NextResponse.json({ 
    status: 'AI Coach API is running',
    model: 'claude-3-5-haiku-20241022',
    timestamp: new Date().toISOString()
  })
}