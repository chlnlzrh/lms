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

  async parseMarkdownToHtml(markdown: string): Promise<string> {
    const result = await this.processor.process(markdown)
    return result.toString()
  }

  calculateReadTime(content: string): number {
    const wordsPerMinute = 200
    const wordCount = content.split(/\s+/).length
    return Math.ceil(wordCount / wordsPerMinute)
  }

  extractLessonMetadata(filename: string, trackFromPath?: string | null): {
    track: string
    moduleNumber?: number
    lessonNumber?: number
    slug: string
  } {
    // Pattern 1: M00-L001-topic-name--date.md (for DE/SaaS tracks with dates)
    const dateMatch = filename.match(/^M(\d+)-L(\d+)-(.+?)--\d{4}-\d{2}-\d{2}\.md$/)
    if (dateMatch) {
      return {
        track: trackFromPath || 'data-engineering',
        moduleNumber: parseInt(dateMatch[1]),
        lessonNumber: parseInt(dateMatch[2]),
        slug: dateMatch[3]
      }
    }

    // Pattern 2: M01-L001-topic-name.md (for AI and other tracks)
    const simpleMatch = filename.match(/^M(\d+)-L(\d+)-(.+)\.md$/)
    if (simpleMatch) {
      return {
        track: trackFromPath || 'ai',
        moduleNumber: parseInt(simpleMatch[1]),
        lessonNumber: parseInt(simpleMatch[2]),
        slug: simpleMatch[3]
      }
    }

    // Pattern 3: M6-L001.md (simple module-lesson format)
    const basicMatch = filename.match(/^M(\d+)-L(\d+)\.md$/)
    if (basicMatch) {
      return {
        track: trackFromPath || 'unknown',
        moduleNumber: parseInt(basicMatch[1]),
        lessonNumber: parseInt(basicMatch[2]),
        slug: `M${basicMatch[1]}-L${basicMatch[2]}`
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
      
      // Extract track from file path
      const pathParts = filePath.split(path.sep)
      const dataIndex = pathParts.findIndex(part => part === 'data')
      const trackFromPath = dataIndex !== -1 && dataIndex < pathParts.length - 2 ? pathParts[dataIndex + 1] : null
      
      const metadata = this.extractLessonMetadata(filename, trackFromPath)
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
      const track = filePath.includes('/ai/') ? 'ai' : filePath.includes('/de/') ? 'data-engineering' : filePath.includes('/saas/') ? 'saas' : 'data-engineering'
      
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
        track: track as 'ai' | 'data-engineering' | 'saas',
        moduleNumber
      }
    } catch (error) {
      console.error(`Error parsing module description ${filePath}:`, error)
      return null
    }
  }

  async getAllLessons(track: string): Promise<ParsedLesson[]> {
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

  async getAllModules(track: string): Promise<ModuleDescription[]> {
    const trackDir = this.getTrackDirectory(track)
    
    // Try to load from modules-descriptions JSON file first for all tracks
    const moduleDescJsonPath = path.join(CONTENT_BASE_PATH, trackDir, 'modules-descriptions', 'module.json')
    if (fs.existsSync(moduleDescJsonPath)) {
      try {
        const jsonContent = fs.readFileSync(moduleDescJsonPath, 'utf-8')
        const data = JSON.parse(jsonContent)
        const modules = data.modules || data
        
        return modules.map((module: any, index: number) => ({
          id: module.id || `${track}-module-${index}`,
          title: module.title,
          description: module.subtitle || module.description,
          duration: module.duration,
          lessonCount: module.lessons || module.lessonCount,
          labCount: module.labs || module.labCount || 0,
          prerequisites: module.prerequisites || [],
          learningObjectives: module.skillsGained || module.learningObjectives || [],
          topics: module.keyTopics || module.topics || [],
          track: track,
          moduleNumber: module.id ? parseInt(module.id.split('-').pop() || '0') : index
        }))
      } catch (error) {
        console.error(`Error loading ${track} modules from modules-descriptions JSON:`, error)
      }
    }

    // For SaaS track, also try the old path
    if (track === 'saas') {
      
      // Fallback to the other JSON file
      const jsonPath = path.join(CONTENT_BASE_PATH, trackDir, 'module.json')
      if (fs.existsSync(jsonPath)) {
        try {
          const jsonContent = fs.readFileSync(jsonPath, 'utf-8')
          const data = JSON.parse(jsonContent)
          const modules = data.modules || data
          
          return modules.map((module: any, index: number) => ({
            id: `saas-module-${module.moduleNumber}`,
            title: module.title,
            description: module.description,
            duration: module.duration,
            lessonCount: module.lessonCount,
            labCount: module.labCount || 0,
            prerequisites: module.prerequisites || [],
            learningObjectives: module.learningObjectives || [],
            topics: module.skillsGained || [],
            track: 'saas' as const,
            moduleNumber: module.moduleNumber
          }))
        } catch (error) {
          console.error(`Error loading SaaS modules from JSON:`, error)
        }
      }
    }
    
    // First try to get modules from description files
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

  async generateModulesFromLessons(track: string): Promise<ModuleDescription[]> {
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

    const moduleTitles = track === 'ai' ? aiModuleTitles : track === 'saas' ? {} : deModuleTitles

    // Generate module descriptions
    const modules: ModuleDescription[] = Object.entries(moduleGroups)
      .map(([moduleNumStr, moduleLessons]) => {
        const moduleNumber = parseInt(moduleNumStr)
        const title = (moduleTitles as any)[moduleNumber] || `Module ${moduleNumber}`
        
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
          track: track,
          moduleNumber
        }
      })
      .sort((a, b) => a.moduleNumber - b.moduleNumber)

    return modules
  }

  async getTrackInfo(track: string): Promise<TrackInfo> {
    const trackDir = this.getTrackDirectory(track)
    const [lessons, modules] = await Promise.all([
      this.getAllLessons(track),
      this.getAllModules(track)
    ])

    // Try to get track metadata from module.json file
    const moduleDescJsonPath = path.join(CONTENT_BASE_PATH, trackDir, 'modules-descriptions', 'module.json')
    let trackMetadata: any = null
    
    if (fs.existsSync(moduleDescJsonPath)) {
      try {
        const jsonContent = fs.readFileSync(moduleDescJsonPath, 'utf-8')
        const data = JSON.parse(jsonContent)
        trackMetadata = data.track
      } catch (error) {
        console.error(`Error loading track metadata for ${track}:`, error)
      }
    }

    // Fallback track titles and descriptions
    const trackTitles: { [key: string]: string } = {
      ai: 'Artificial Intelligence',
      'data-engineering': 'Data Engineering',
      de: 'Data Engineering',
      saas: 'SaaS Development',
      sfdc: 'Salesforce',
      snowflake_tune: 'Snowflake Tuning',
      workato: 'Workato',
      ba: 'Business Analyst',
      data_engineer: 'Data Engineer',
      data_gov: 'Data Governance',
      devops_engineer: 'DevOps Engineer',
      finance: 'Finance',
      hr: 'Human Resources',
      mdm: 'Master Data Management',
      pm: 'Project Manager',
      qa: 'Quality Assurance',
      rpa: 'Robotic Process Automation',
      sales: 'Sales',
      sfdc_engineer: 'Salesforce Engineer',
      ta: 'Talent Acquisition',
      viz_engineer: 'Visualization Engineer',
      workato_engineer: 'Workato Engineer'
    }

    const trackDescriptions: { [key: string]: string } = {
      ai: 'Master AI tools, prompt engineering, and modern AI development workflows',
      'data-engineering': 'Learn data warehousing, SQL, ETL/ELT, and modern data engineering practices',
      de: 'Learn data warehousing, SQL, ETL/ELT, and modern data engineering practices',
      saas: 'Build scalable SaaS applications with modern architecture patterns and best practices',
      sfdc: 'Master Salesforce development, administration, and platform customization',
      snowflake_tune: 'Optimize Snowflake performance with advanced tuning and best practices',
      workato: 'Build powerful integrations and automation workflows with Workato platform',
      ba: 'Develop business analysis skills and requirements gathering expertise',
      data_engineer: 'Master data engineering principles, pipelines, and infrastructure',
      data_gov: 'Implement data governance frameworks and ensure data quality compliance',
      devops_engineer: 'Learn DevOps practices, CI/CD, and infrastructure automation',
      finance: 'Apply technology solutions to financial processes and systems',
      hr: 'Leverage technology for human resources management and operations',
      mdm: 'Master data management strategies, governance, and quality assurance',
      pm: 'Develop project management skills with modern methodologies and tools',
      qa: 'Master quality assurance practices, testing frameworks, and automation',
      rpa: 'Build robotic process automation solutions for business efficiency',
      sales: 'Apply technology and analytics to sales processes and customer management',
      sfdc_engineer: 'Engineer advanced Salesforce solutions and platform integrations',
      ta: 'Master talent acquisition strategies, recruitment technologies, and candidate assessment',
      viz_engineer: 'Create compelling data visualizations and business intelligence solutions',
      workato_engineer: 'Engineer complex integration solutions using Workato platform'
    }

    // Use metadata from JSON file if available, otherwise fallback to hardcoded values
    const title = trackMetadata?.title || trackTitles[track] || track.charAt(0).toUpperCase() + track.slice(1)
    const description = trackMetadata?.description || trackDescriptions[track] || `Learn ${track} concepts and best practices`
    const totalLessons = trackMetadata?.lessons || lessons.length
    const totalModules = trackMetadata?.modules || modules.length
    const estimatedDuration = trackMetadata?.duration || `${totalModules} weeks`

    return {
      id: track,
      title,
      description,
      totalLessons,
      totalModules,
      estimatedDuration,
      modules
    }
  }
}

export const contentParser = new ContentParser()