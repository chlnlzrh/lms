import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { remark } from 'remark'
import remarkHtml from 'remark-html'
import remarkGfm from 'remark-gfm'
import { ParsedLesson, LessonFrontmatter, ModuleDescription, TrackInfo } from '@/types/content'

const CONTENT_BASE_PATH = path.join(process.cwd(), '..', '..', 'ai', 'training')

class ContentParser {
  private processor = remark().use(remarkGfm).use(remarkHtml)

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
    // AI lessons: M01-L001-topic-name.md
    const aiMatch = filename.match(/^M(\d+)-L(\d+)-(.+)\.md$/)
    if (aiMatch) {
      return {
        track: 'ai',
        moduleNumber: parseInt(aiMatch[1]),
        lessonNumber: parseInt(aiMatch[2]),
        slug: aiMatch[3]
      }
    }

    // DE lessons: Topic-Area--specific-topic--date.md
    const deMatch = filename.match(/^(.+?)--(.+?)--\d{4}-\d{2}-\d{2}\.md$/)
    if (deMatch) {
      return {
        track: 'data-engineering',
        slug: `${deMatch[1]}-${deMatch[2]}`
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
      const track = filePath.includes('/ai/') ? 'ai' : 'data-engineering'
      
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
    const lessonsPath = path.join(CONTENT_BASE_PATH, track, 'lessons')
    
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

    return lessons.filter(lesson => lesson !== null) as ParsedLesson[]
  }

  async getAllModules(track: 'ai' | 'data-engineering'): Promise<ModuleDescription[]> {
    const modulesPath = path.join(CONTENT_BASE_PATH, track, 'modules-descriptions')
    
    if (!fs.existsSync(modulesPath)) {
      console.warn(`Modules directory not found: ${modulesPath}`)
      return []
    }

    const files = fs.readdirSync(modulesPath)
      .filter(file => file.endsWith('.md'))
      .filter(file => !file.startsWith('.'))

    const modules = await Promise.all(
      files.map(file => this.parseModuleDescription(path.join(modulesPath, file)))
    )

    return modules
      .filter(module => module !== null)
      .sort((a, b) => a!.moduleNumber - b!.moduleNumber) as ModuleDescription[]
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