import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { remark } from 'remark'
import remarkHtml from 'remark-html'
import remarkGfm from 'remark-gfm'
import { ParsedLesson, LessonFrontmatter, ModuleDescription, TrackInfo } from '@/types/content'

const CONTENT_BASE_PATH = path.join(process.cwd(), 'src', 'data')

class ContentParser {
  private processor = remark().use(remarkGfm).use(remarkHtml)

  private getTrackDirectory(track: 'ai' | 'data-engineering'): string {
    const trackDirectoryMap = {
      'ai': 'ai',
      'data-engineering': 'de'
    }
    return trackDirectoryMap[track]
  }

  async parseMarkdownToHtml(markdown: string): Promise<string> {
    const result = await this.processor.process(markdown)
    return result.toString()
  }

  calculateReadTime(content: string): number {
    const wordsPerMinute = 200
    const wordCount = content.split(/\s+/).length
    return Math.ceil(wordCount / wordsPerMinute)
  }

  extractLessonMetadata(filename: string): {
    track: 'ai' | 'data-engineering'
    moduleNumber?: number
    lessonNumber?: number
    slug: string
  } {
    // DE lessons: M01-L001-topic-name--date.md (check first - more specific)
    const deMatch = filename.match(/^M(\d+)-L(\d+)-(.+?)--\d{4}-\d{2}-\d{2}\.md$/)
    if (deMatch) {
      return {
        track: 'data-engineering',
        moduleNumber: parseInt(deMatch[1]),
        lessonNumber: parseInt(deMatch[2]),
        slug: deMatch[3]
      }
    }

    // AI lessons: M01-L001-topic-name.md (check second - more general)
    const aiMatch = filename.match(/^M(\d+)-L(\d+)-(.+)\.md$/)
    if (aiMatch) {
      return {
        track: 'ai',
        moduleNumber: parseInt(aiMatch[1]),
        lessonNumber: parseInt(aiMatch[2]),
        slug: aiMatch[3]
      }
    }

    // Legacy DE lessons: Topic-Area--specific-topic--date.md
    const legacyDeMatch = filename.match(/^(.+?)--(.+?)--\d{4}-\d{2}-\d{2}\.md$/)
    if (legacyDeMatch) {
      return {
        track: 'data-engineering',
        slug: `${legacyDeMatch[1]}-${legacyDeMatch[2]}`
      }
    }

    // Fallback
    return {
      track: 'data-engineering',
      slug: filename.replace('.md', '')
    }
  }

  async parseLesson(filePath: string): Promise<ParsedLesson | null> {
    try {
      const filename = path.basename(filePath)
      const fileContent = fs.readFileSync(filePath, 'utf-8')
      const { data: frontmatter, content } = matter(fileContent)
      
      const metadata = this.extractLessonMetadata(filename)
      const htmlContent = await this.parseMarkdownToHtml(content)
      const estimatedReadTime = this.calculateReadTime(content)
      
      // Extract title from content if not in frontmatter
      const title = frontmatter.title || 
                   content.split('\n')[0]?.replace(/^#\s*/, '') || 
                   metadata.slug.replace(/-/g, ' ')

      return {
        id: `${metadata.track}-${metadata.slug}`,
        slug: metadata.slug,
        frontmatter: {
          title,
          module: frontmatter.module,
          duration: frontmatter.duration,
          complexity: frontmatter.complexity,
          topics: frontmatter.topics || [],
          prerequisites: frontmatter.prerequisites || [],
          learningObjectives: frontmatter.learningObjectives || []
        },
        content,
        htmlContent,
        filePath,
        track: metadata.track,
        moduleNumber: metadata.moduleNumber,
        lessonNumber: metadata.lessonNumber,
        estimatedReadTime
      }
    } catch (error) {
      console.error(`Error parsing lesson ${filePath}:`, error)
      return null
    }
  }

  async parseModuleDescription(filePath: string): Promise<ModuleDescription | null> {
    try {
      const filename = path.basename(filePath)
      const fileContent = fs.readFileSync(filePath, 'utf-8')
      const { data: frontmatter, content } = matter(fileContent)
      
      // Extract module number and track from filename
      const moduleMatch = filename.match(/module-(\d+)-(.+?)--(\d{4}-\d{2}-\d{2})\.md$/)
      if (!moduleMatch) return null
      
      const moduleNumber = parseInt(moduleMatch[1])
      const track = filePath.includes('/ai/') ? 'ai' : filePath.includes('/de/') ? 'data-engineering' : 'data-engineering'
      
      // Parse content for structured information
      const durationMatch = content.match(/\*\*Duration:\*\*\s*(.+)/i)
      const lessonMatch = content.match(/\*\*Lessons:\*\*\s*(\d+)\s*lessons/i)
      const labMatch = content.match(/\*\*Labs:\*\*\s*(\d+)\s*labs/i)
      
      const title = content.split('\n')[0]?.replace(/^#\s*/, '') || 
                   frontmatter.title || 
                   `Module ${moduleNumber}`
      
      const description = content.split('\n')[2] || ''

      return {
        id: `${track}-module-${moduleNumber}`,
        title,
        description,
        duration: durationMatch?.[1] || '0 hours',
        lessonCount: parseInt(lessonMatch?.[1] || '0'),
        labCount: parseInt(labMatch?.[1] || '0'),
        prerequisites: frontmatter.prerequisites || [],
        learningObjectives: frontmatter.learningObjectives || [],
        topics: frontmatter.topics || [],
        track: track as 'ai' | 'data-engineering',
        moduleNumber
      }
    } catch (error) {
      console.error(`Error parsing module description ${filePath}:`, error)
      return null
    }
  }

  async getAllLessons(track: 'ai' | 'data-engineering'): Promise<ParsedLesson[]> {
    const trackDir = this.getTrackDirectory(track)
    const lessonsPath = path.join(CONTENT_BASE_PATH, trackDir, 'lessons')
    
    if (!fs.existsSync(lessonsPath)) {
      console.warn(`Lessons directory not found: ${lessonsPath}`)
      return []
    }

    const files = fs.readdirSync(lessonsPath)
      .filter(file => file.endsWith('.md'))
      .filter(file => !file.startsWith('.'))

    const lessons = await Promise.all(
      files.map(file => this.parseLesson(path.join(lessonsPath, file)))
    )

    const validLessons = lessons.filter(lesson => lesson !== null) as ParsedLesson[]

    // Sort lessons by module number, then by lesson number
    return validLessons.sort((a, b) => {
      // First sort by module number
      if (a.moduleNumber && b.moduleNumber && a.moduleNumber !== b.moduleNumber) {
        return a.moduleNumber - b.moduleNumber
      }
      
      // Then sort by lesson number within the same module
      if (a.lessonNumber && b.lessonNumber) {
        return a.lessonNumber - b.lessonNumber
      }
      
      // Fallback to alphabetical sorting if no numbers available
      return a.slug.localeCompare(b.slug)
    })
  }

  async getAllModules(track: 'ai' | 'data-engineering'): Promise<ModuleDescription[]> {
    // First try to get modules from description files
    const trackDir = this.getTrackDirectory(track)
    const modulesPath = path.join(CONTENT_BASE_PATH, trackDir, 'modules-descriptions')
    
    if (fs.existsSync(modulesPath)) {
      const files = fs.readdirSync(modulesPath)
        .filter(file => file.endsWith('.md'))
        .filter(file => !file.startsWith('.'))

      if (files.length > 0) {
        const modules = await Promise.all(
          files.map(file => this.parseModuleDescription(path.join(modulesPath, file)))
        )

        return modules
          .filter(module => module !== null)
          .sort((a, b) => a!.moduleNumber - b!.moduleNumber) as ModuleDescription[]
      }
    }

    // If no module descriptions found, generate modules dynamically from lessons
    console.log(`No module descriptions found for ${track}, generating from lessons...`)
    return this.generateModulesFromLessons(track)
  }

  async generateModulesFromLessons(track: 'ai' | 'data-engineering'): Promise<ModuleDescription[]> {
    const lessons = await this.getAllLessons(track)
    
    // Group lessons by module number
    const moduleGroups: { [moduleNumber: number]: ParsedLesson[] } = {}
    
    lessons.forEach(lesson => {
      if (lesson.moduleNumber) {
        if (!moduleGroups[lesson.moduleNumber]) {
          moduleGroups[lesson.moduleNumber] = []
        }
        moduleGroups[lesson.moduleNumber].push(lesson)
      }
    })

    // Content Structure mappings for module titles
    const aiModuleTitles = {
      1: 'AI Foundation & Tool Fluency',
      2: 'AI in the Software Development Lifecycle', 
      3: 'AI-Augmented Engineering (Role-Specific)',
      4: 'AI Agent & Platform Architecture',
      5: 'AI Strategy, Delivery & Governance',
      6: 'Continuous Learning & Innovation Culture'
    }

    const deModuleTitles = {
      1: 'Data Engineering Foundations',
      2: 'SQL & Data Modeling',
      3: 'Python for Data Engineering',
      4: 'Modern Warehouse & Transformation',
      5: 'Batch Processing & Cloud Platforms', 
      6: 'Orchestration & DataOps',
      7: 'Real-Time Data & Streaming'
    }

    const moduleTitles = track === 'ai' ? aiModuleTitles : deModuleTitles

    // Generate module descriptions
    const modules: ModuleDescription[] = Object.entries(moduleGroups)
      .map(([moduleNumStr, moduleLessons]) => {
        const moduleNumber = parseInt(moduleNumStr)
        const title = moduleTitles[moduleNumber as keyof typeof moduleTitles] || `Module ${moduleNumber}`
        
        // Calculate estimated duration (assume 5 minutes per lesson)
        const estimatedMinutes = moduleLessons.length * 5
        const hours = Math.ceil(estimatedMinutes / 60)
        
        return {
          id: `${track}-module-${moduleNumber}`,
          title,
          description: `Learn key concepts and practical skills in ${title.toLowerCase()}`,
          duration: `${hours} hours`,
          lessonCount: moduleLessons.length,
          labCount: 0,
          prerequisites: [],
          learningObjectives: [],
          topics: [],
          track: track as 'ai' | 'data-engineering',
          moduleNumber
        }
      })
      .sort((a, b) => a.moduleNumber - b.moduleNumber)

    return modules
  }

  async getTrackInfo(track: 'ai' | 'data-engineering'): Promise<TrackInfo> {
    const [lessons, modules] = await Promise.all([
      this.getAllLessons(track),
      this.getAllModules(track)
    ])

    const totalDuration = modules.reduce((total, module) => {
      const hours = parseInt(module.duration.match(/(\d+)/)?.[1] || '0')
      return total + hours
    }, 0)

    const trackTitles = {
      ai: 'AI Training Track',
      'data-engineering': 'Data Engineering Track'
    }

    const trackDescriptions = {
      ai: 'Master AI tools, prompt engineering, and modern AI development workflows',
      'data-engineering': 'Learn data warehousing, SQL, ETL/ELT, and modern data engineering practices'
    }

    return {
      id: track,
      title: trackTitles[track],
      description: trackDescriptions[track],
      totalLessons: lessons.length,
      totalModules: modules.length,
      estimatedDuration: `${totalDuration} hours`,
      modules
    }
  }
}

export const contentParser = new ContentParser()