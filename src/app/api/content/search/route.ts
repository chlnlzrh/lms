import { NextRequest, NextResponse } from 'next/server'
import { contentParser } from '@/lib/content-parser'

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const query = searchParams.get('q') || ''
    const track = searchParams.get('track') as 'ai' | 'data-engineering' | null
    const type = searchParams.get('type') as 'lesson' | 'module' | null
    const complexity = searchParams.get('complexity') as 'F' | 'I' | 'A' | null

    if (query.length < 2) {
      return NextResponse.json([])
    }

    // Load content
    const [aiTrack, deTrack] = await Promise.all([
      contentParser.getTrackInfo('ai'),
      contentParser.getTrackInfo('data-engineering')
    ])

    const [aiLessons, deLessons] = await Promise.all([
      contentParser.getAllLessons('ai'),
      contentParser.getAllLessons('data-engineering')
    ])

    const allLessons = [...aiLessons, ...deLessons]
    const allModules = [...aiTrack.modules, ...deTrack.modules]

    // Build search index
    const searchIndex = []

    // Index lessons
    for (const lesson of allLessons) {
      searchIndex.push({
        id: lesson.id,
        title: lesson.frontmatter.title,
        content: lesson.content.substring(0, 1000),
        track: lesson.track,
        module: lesson.frontmatter.module || `Module ${lesson.moduleNumber || 'Unknown'}`,
        type: 'lesson' as const,
        keywords: [
          ...lesson.frontmatter.topics || [],
          lesson.track,
          lesson.frontmatter.complexity || '',
          lesson.slug.split('-')
        ].filter(Boolean)
      })
    }

    // Index modules
    for (const module of allModules) {
      searchIndex.push({
        id: module.id,
        title: module.title,
        content: module.description,
        track: module.track,
        module: module.title,
        type: 'module' as const,
        keywords: [
          ...module.topics,
          module.track,
          `module-${module.moduleNumber}`,
          ...module.learningObjectives
        ].filter(Boolean)
      })
    }

    // Filter and search
    const searchTerms = query.toLowerCase().split(' ').filter(term => term.length > 1)
    
    const results = searchIndex.filter(entry => {
      // Apply filters
      if (track && entry.track !== track) return false
      if (type && entry.type !== type) return false
      
      // Search in title, content, and keywords
      const searchText = `${entry.title} ${entry.content} ${entry.keywords.join(' ')}`.toLowerCase()
      
      return searchTerms.some(term => searchText.includes(term))
    }).slice(0, 50) // Limit results for performance

    return NextResponse.json(results)
  } catch (error) {
    console.error('Search error:', error)
    return NextResponse.json(
      { error: 'Search failed' },
      { status: 500 }
    )
  }
}