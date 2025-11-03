import fs from 'fs'
import path from 'path'

const CONTENT_BASE_PATH = path.join(process.cwd(), 'src', 'data')

interface FastTrackInfo {
  id: string
  title: string
  description: string
  totalLessons: number
  totalModules: number
  estimatedDuration: string
  modules: FastModuleInfo[]
  type: 'Book of Knowledge' | 'Learning Path'
  stats?: {
    estimatedHours: number
    practicalRatio: number
    handsOnLessons: number
  }
}

interface FastModuleInfo {
  id: string
  title: string
  description: string
  duration: string
  lessonCount: number
  labCount: number
  prerequisites: string[]
  learningObjectives: string[]
  topics: string[]
  moduleNumber: number
}

interface LandingPageData {
  pageInfo: {
    title: string
    description: string
  }
  overview: {
    totals: {
      tracks: number
      lessons: number
      estimatedHours: number
    }
  }
  tracks: Array<{
    id: string
    title: string
    description: string
    icon: string
    color: string
    stats: {
      modules: number
      lessons: number
      duration: string
      estimatedHours: number
      practicalRatio: number
      handsOnLessons: number
    }
    keyTopics: string[]
    difficulty: string
  }>
}

class FastContentParser {
  private landingPageCache: { [key: string]: LandingPageData } = {}
  private moduleCache: { [key: string]: any } = {}

  private getTrackDirectory(track: string): string {
    const trackDirectoryMap: { [key: string]: string } = {
      'ai': 'ai',
      'data-engineering': 'de', 
      'de': 'de',
      'saas': 'saas',
      'sfdc': 'sfdc',
      'snowflake_tune': 'snowflake_tune',
      'workato': 'workato',
      'ba': 'ba',
      'data_engineer': 'data_engineer',
      'data_gov': 'data_gov',
      'devops_engineer': 'devops_engineer',
      'finance': 'finance',
      'hr': 'hr',
      'mdm': 'mdm',
      'pm': 'pm',
      'qa': 'qa',
      'rpa': 'rpa',
      'sales': 'sales',
      'sfdc_engineer': 'sfdc_engineer',
      'ta': 'ta',
      'viz_engineer': 'viz_engineer',
      'workato_engineer': 'workato_engineer'
    }
    return trackDirectoryMap[track] || track
  }

  private loadLandingPageData(type: 'Book of Knowledge' | 'Learning Path'): LandingPageData | null {
    const cacheKey = type.toLowerCase().replace(' ', '-')
    
    if (this.landingPageCache[cacheKey]) {
      return this.landingPageCache[cacheKey]
    }

    try {
      const filename = type === 'Book of Knowledge' 
        ? 'book-of-knowledge-landing.json'
        : 'learning-path-landing.json'
      
      const filePath = path.join(CONTENT_BASE_PATH, filename)
      
      if (!fs.existsSync(filePath)) {
        console.warn(`Landing page data not found: ${filePath}`)
        return null
      }

      const jsonContent = fs.readFileSync(filePath, 'utf-8')
      const data = JSON.parse(jsonContent)
      
      this.landingPageCache[cacheKey] = data
      return data
    } catch (error) {
      console.error(`Error loading landing page data for ${type}:`, error)
      return null
    }
  }

  private loadModuleData(trackId: string): any | null {
    if (this.moduleCache[trackId]) {
      return this.moduleCache[trackId]
    }

    try {
      const trackDir = this.getTrackDirectory(trackId)
      const modulePath = path.join(CONTENT_BASE_PATH, trackDir, 'modules-descriptions', 'module.json')
      
      if (!fs.existsSync(modulePath)) {
        console.warn(`Module data not found for track: ${trackId}`)
        return null
      }

      const jsonContent = fs.readFileSync(modulePath, 'utf-8')
      const data = JSON.parse(jsonContent)
      
      this.moduleCache[trackId] = data
      return data
    } catch (error) {
      console.error(`Error loading module data for ${trackId}:`, error)
      return null
    }
  }

  async getTrackInfo(trackId: string): Promise<FastTrackInfo | null> {
    // First, try to get track info from landing page data (fastest)
    const bookData = this.loadLandingPageData('Book of Knowledge')
    const learningData = this.loadLandingPageData('Learning Path')
    
    let trackData = null
    let trackType: 'Book of Knowledge' | 'Learning Path' = 'Learning Path'
    
    // Check Book of Knowledge tracks first
    if (bookData) {
      trackData = bookData.tracks.find(track => track.id === trackId)
      if (trackData) {
        trackType = 'Book of Knowledge'
      }
    }
    
    // If not found, check Learning Path tracks
    if (!trackData && learningData) {
      trackData = learningData.tracks.find(track => track.id === trackId)
      if (trackData) {
        trackType = 'Learning Path'
      }
    }
    
    if (!trackData) {
      console.warn(`Track ${trackId} not found in landing page data`)
      return null
    }
    
    // Get detailed module information from the track's module.json
    const moduleData = this.loadModuleData(trackId)
    let modules: FastModuleInfo[] = []
    
    if (moduleData?.modules) {
      modules = moduleData.modules.map((module: any, index: number) => ({
        id: module.id || `${trackId}-module-${index}`,
        title: module.title || `Module ${index + 1}`,
        description: module.subtitle || module.description || '',
        duration: module.duration || '1 week',
        lessonCount: module.lessons || 0,
        labCount: module.labs || 0,
        prerequisites: module.prerequisites || [],
        learningObjectives: module.skillsGained || [],
        topics: module.keyTopics || [],
        moduleNumber: module.moduleNumber || index
      }))
    }
    
    return {
      id: trackId,
      title: trackData.title,
      description: trackData.description,
      totalLessons: trackData.stats.lessons,
      totalModules: trackData.stats.modules,
      estimatedDuration: trackData.stats.duration,
      type: trackType,
      modules,
      stats: {
        estimatedHours: trackData.stats.estimatedHours,
        practicalRatio: trackData.stats.practicalRatio,
        handsOnLessons: trackData.stats.handsOnLessons
      }
    }
  }

  async getAllBookOfKnowledgeTracks(): Promise<LandingPageData | null> {
    return this.loadLandingPageData('Book of Knowledge')
  }

  async getAllLearningPathTracks(): Promise<LandingPageData | null> {
    return this.loadLandingPageData('Learning Path')
  }

  // Compatibility method for existing code
  async getTrackType(trackId: string): Promise<'Book of Knowledge' | 'Learning Path' | null> {
    const trackInfo = await this.getTrackInfo(trackId)
    return trackInfo?.type || null
  }
}

export const fastContentParser = new FastContentParser()