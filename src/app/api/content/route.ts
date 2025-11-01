import { NextResponse } from 'next/server'
import { contentParser } from '@/lib/content-parser'

export async function GET() {
  try {
    // Load both tracks in parallel to utilize multi-core system
    const [aiTrack, deTrack] = await Promise.all([
      contentParser.getTrackInfo('ai'),
      contentParser.getTrackInfo('data-engineering')
    ])

    // Load all lessons in parallel
    const [aiLessons, deLessons] = await Promise.all([
      contentParser.getAllLessons('ai'),
      contentParser.getAllLessons('data-engineering')
    ])

    const allLessons = [...aiLessons, ...deLessons]
    const allModules = [...aiTrack.modules, ...deTrack.modules]
    const tracks = [aiTrack, deTrack]

    // Build search index
    const searchIndex = allLessons.map(lesson => ({
      id: lesson.id,
      title: lesson.frontmatter.title,
      content: lesson.content.substring(0, 1000), // First 1000 chars for search
      track: lesson.track,
      module: lesson.frontmatter.module || `Module ${lesson.moduleNumber || 'Unknown'}`,
      type: 'lesson' as const,
      keywords: [
        ...lesson.frontmatter.topics || [],
        lesson.track,
        lesson.frontmatter.complexity || '',
        lesson.slug.split('-')
      ].filter(Boolean)
    }))

    // Add modules to search index
    allModules.forEach(module => {
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
    })

    return NextResponse.json({
      lessons: allLessons,
      modules: allModules,
      tracks,
      searchIndex
    })
  } catch (error) {
    console.error('Error loading content:', error)
    return NextResponse.json(
      { error: 'Failed to load content' },
      { status: 500 }
    )
  }
}