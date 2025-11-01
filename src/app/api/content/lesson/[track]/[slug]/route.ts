import { NextRequest, NextResponse } from 'next/server'
import { contentParser } from '@/lib/content-parser'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ track: string; slug: string }> }
) {
  try {
    const resolvedParams = await params
    const { track, slug } = resolvedParams
    
    if (track !== 'ai' && track !== 'data-engineering') {
      return NextResponse.json(
        { error: 'Invalid track' },
        { status: 400 }
      )
    }

    // Get all lessons for the track
    const lessons = await contentParser.getAllLessons(track as 'ai' | 'data-engineering')
    
    // Find the lesson by slug
    const lesson = lessons.find(l => l.slug === slug)
    
    if (!lesson) {
      return NextResponse.json(
        { error: 'Lesson not found' },
        { status: 404 }
      )
    }

    // Get navigation (previous/next lessons)
    const currentIndex = lessons.findIndex(l => l.slug === slug)
    const previousLesson = currentIndex > 0 ? lessons[currentIndex - 1] : null
    const nextLesson = currentIndex < lessons.length - 1 ? lessons[currentIndex + 1] : null

    return NextResponse.json({
      lesson,
      previousLesson,
      nextLesson
    })
  } catch (error) {
    console.error('Error loading lesson:', error)
    return NextResponse.json(
      { error: 'Failed to load lesson' },
      { status: 500 }
    )
  }
}