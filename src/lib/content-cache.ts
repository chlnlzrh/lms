import { ParsedLesson, ModuleDescription, TrackInfo, ContentIndex, SearchIndexEntry } from '@/types/content'

class ContentCache {
  private static instance: ContentCache
  private cache: ContentIndex | null = null
  private isLoading = false
  private loadPromise: Promise<ContentIndex> | null = null

  static getInstance(): ContentCache {
    if (!ContentCache.instance) {
      ContentCache.instance = new ContentCache()
    }
    return ContentCache.instance
  }

  async getContent(): Promise<ContentIndex> {
    if (this.cache) {
      return this.cache
    }

    if (this.isLoading && this.loadPromise) {
      return this.loadPromise
    }

    this.isLoading = true
    this.loadPromise = this.loadContentFromAPI()

    try {
      this.cache = await this.loadPromise
      return this.cache
    } finally {
      this.isLoading = false
      this.loadPromise = null
    }
  }

  private async loadContentFromAPI(): Promise<ContentIndex> {
    console.log('Loading content from API...')
    const startTime = Date.now()

    try {
      const response = await fetch('/api/content')
      if (!response.ok) {
        throw new Error(`Failed to fetch content: ${response.statusText}`)
      }
      
      const content = await response.json()
      
      const loadTime = Date.now() - startTime
      console.log(`Content loaded in ${loadTime}ms`)
      console.log(`- ${content.lessons.length} lessons`)
      console.log(`- ${content.modules.length} modules`)
      console.log(`- ${content.searchIndex.length} search entries`)

      return content
    } catch (error) {
      console.error('Failed to load content from API:', error)
      throw error
    }
  }


  // Utility methods for quick access
  async getLessonsByTrack(track: 'ai' | 'data-engineering'): Promise<ParsedLesson[]> {
    const content = await this.getContent()
    return content.lessons.filter(lesson => lesson.track === track)
  }

  async getModulesByTrack(track: 'ai' | 'data-engineering'): Promise<ModuleDescription[]> {
    const content = await this.getContent()
    return content.modules.filter(module => module.track === track)
  }

  async getLessonById(id: string): Promise<ParsedLesson | null> {
    const content = await this.getContent()
    return content.lessons.find(lesson => lesson.id === id) || null
  }

  async getLessonBySlug(track: 'ai' | 'data-engineering', slug: string): Promise<{
    lesson: ParsedLesson
    previousLesson?: ParsedLesson
    nextLesson?: ParsedLesson
  } | null> {
    try {
      const response = await fetch(`/api/content/lesson/${track}/${slug}`)
      if (!response.ok) {
        if (response.status === 404) return null
        throw new Error(`Failed to fetch lesson: ${response.statusText}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching lesson:', error)
      return null
    }
  }

  async getModuleById(id: string): Promise<ModuleDescription | null> {
    const content = await this.getContent()
    return content.modules.find(module => module.id === id) || null
  }

  async searchContent(query: string, filters?: {
    track?: 'ai' | 'data-engineering'
    type?: 'lesson' | 'module'
    complexity?: 'F' | 'I' | 'A'
  }): Promise<SearchIndexEntry[]> {
    try {
      const params = new URLSearchParams({
        q: query,
        ...(filters?.track && { track: filters.track }),
        ...(filters?.type && { type: filters.type }),
        ...(filters?.complexity && { complexity: filters.complexity })
      })

      const response = await fetch(`/api/content/search?${params}`)
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Search error:', error)
      return []
    }
  }

  // Force refresh cache
  async refresh(): Promise<ContentIndex> {
    this.cache = null
    return this.getContent()
  }
}

export const contentCache = ContentCache.getInstance()