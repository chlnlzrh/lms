export interface LessonFrontmatter {
  title: string
  module?: string
  duration?: string
  complexity?: 'F' | 'I' | 'A' // Foundational, Intermediate, Advanced
  topics?: string[]
  prerequisites?: string[]
  learningObjectives?: string[]
}

export interface ParsedLesson {
  id: string
  slug: string
  frontmatter: LessonFrontmatter
  content: string
  htmlContent: string
  filePath: string
  track: 'ai' | 'data-engineering' | 'saas'
  moduleNumber?: number
  lessonNumber?: number
  estimatedReadTime: number
}

export interface ModuleDescription {
  id: string
  title: string
  description: string
  duration: string
  lessonCount: number
  labCount: number
  prerequisites: string[]
  learningObjectives: string[]
  topics: string[]
  track: 'ai' | 'data-engineering' | 'saas'
  moduleNumber: number
}

export interface TrackInfo {
  id: 'ai' | 'data-engineering' | 'saas'
  title: string
  description: string
  totalLessons: number
  totalModules: number
  estimatedDuration: string
  modules: ModuleDescription[]
}

export interface ContentIndex {
  lessons: ParsedLesson[]
  modules: ModuleDescription[]
  tracks: TrackInfo[]
  searchIndex: SearchIndexEntry[]
}

export interface SearchIndexEntry {
  id: string
  title: string
  content: string
  track: string
  module: string
  type: 'lesson' | 'module'
  keywords: string[]
}