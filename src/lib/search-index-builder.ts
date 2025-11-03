// Server-side only search index builder
import { fastContentParser } from './fast-content-parser'
import type { SearchResult } from './search'

// Build search index from existing content (server-side only)
export async function buildSearchIndex(): Promise<SearchResult[]> {
  const results: SearchResult[] = []

  try {
    // Get all tracks from both learning types
    const [bookTracks, learningTracks] = await Promise.all([
      fastContentParser.getAllBookOfKnowledgeTracks(),
      fastContentParser.getAllLearningPathTracks()
    ])

    // Process Book of Knowledge tracks
    for (const track of bookTracks.tracks) {
      // Add track itself as a result
      results.push({
        id: `book-track-${track.id}`,
        title: track.title,
        description: track.description,
        href: `/book-of-knowledge/${track.id}`,
        track: track.id,
        difficulty: 'beginner' as const,
        duration: track.stats.duration,
        type: 'track' as const,
        keywords: [track.title.toLowerCase(), 'knowledge', 'fundamentals', track.id]
      })

      // Get track details for modules and lessons
      try {
        const trackInfo = await fastContentParser.getTrackInfo(track.id)
        if (trackInfo?.modules) {
          for (const module of trackInfo.modules) {
            // Add module as a result
            results.push({
              id: `book-module-${track.id}-${module.id}`,
              title: module.title,
              description: module.description,
              href: `/book-of-knowledge/${track.id}#${module.id}`,
              track: track.id,
              difficulty: 'intermediate' as const,
              duration: module.duration,
              type: 'module' as const,
              keywords: [module.title.toLowerCase(), 'module', track.title.toLowerCase()]
            })

            // Add lessons if available
            if (module.lessons) {
              for (const lesson of module.lessons) {
                results.push({
                  id: `book-lesson-${track.id}-${module.id}-${lesson.id}`,
                  title: lesson.title,
                  description: lesson.description || `${lesson.title} lesson in ${module.title}`,
                  href: `/book-of-knowledge/${track.id}/lesson/${lesson.id}`,
                  track: track.id,
                  difficulty: lesson.difficulty || 'beginner' as const,
                  duration: lesson.duration || '30 min',
                  type: 'lesson' as const,
                  keywords: [
                    lesson.title.toLowerCase(),
                    module.title.toLowerCase(),
                    track.title.toLowerCase(),
                    'lesson',
                    ...(lesson.topics || [])
                  ]
                })
              }
            }
          }
        }
      } catch (error) {
        console.warn(`Error loading track info for ${track.id}:`, error)
      }
    }

    // Process Learning Path tracks
    for (const track of learningTracks.tracks) {
      // Add track itself as a result
      results.push({
        id: `learning-track-${track.id}`,
        title: track.title,
        description: track.description,
        href: `/learning-path/${track.id}`,
        track: track.id,
        difficulty: 'beginner' as const,
        duration: track.stats.duration,
        type: 'track' as const,
        keywords: [track.title.toLowerCase(), 'learning path', 'career', track.id]
      })

      // Get track details for modules and lessons
      try {
        const trackInfo = await fastContentParser.getTrackInfo(track.id)
        if (trackInfo?.modules) {
          for (const module of trackInfo.modules) {
            // Add module as a result
            results.push({
              id: `learning-module-${track.id}-${module.id}`,
              title: module.title,
              description: module.description,
              href: `/learning-path/${track.id}#${module.id}`,
              track: track.id,
              difficulty: 'intermediate' as const,
              duration: module.duration,
              type: 'module' as const,
              keywords: [module.title.toLowerCase(), 'module', track.title.toLowerCase()]
            })

            // Add lessons if available
            if (module.lessons) {
              for (const lesson of module.lessons) {
                results.push({
                  id: `learning-lesson-${track.id}-${module.id}-${lesson.id}`,
                  title: lesson.title,
                  description: lesson.description || `${lesson.title} lesson in ${module.title}`,
                  href: `/learning-path/${track.id}/lesson/${lesson.id}`,
                  track: track.id,
                  difficulty: lesson.difficulty || 'beginner' as const,
                  duration: lesson.duration || '30 min',
                  type: 'lesson' as const,
                  keywords: [
                    lesson.title.toLowerCase(),
                    module.title.toLowerCase(),
                    track.title.toLowerCase(),
                    'lesson',
                    ...(lesson.topics || [])
                  ]
                })
              }
            }
          }
        }
      } catch (error) {
        console.warn(`Error loading track info for ${track.id}:`, error)
      }
    }

    console.log(`Search index built with ${results.length} items`)
    
  } catch (error) {
    console.error('Error building search index:', error)
  }

  return results
}