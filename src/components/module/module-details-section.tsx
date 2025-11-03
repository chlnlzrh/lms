import fs from 'fs'
import path from 'path'

interface ModuleData {
  id: string
  moduleNumber: number
  title: string
  subtitle: string
  description: string
  duration: string
  estimatedDuration: number
  readingTime: number
  lessons: number
  handsOnLessons: number
  labs: number
  prerequisites: string[]
  keyTopics: string[]
  skillsGained: string[]
  lessonsData?: any[]
  progress?: {
    completed: number
    total: number
    percentage: number
  }
}

interface ModuleDetailsSectionProps {
  moduleNumber: number
  color: string
  track?: string
}

export function ModuleDetailsSection({ moduleNumber, color, track = 'de' }: ModuleDetailsSectionProps) {
  try {
    // Map track names to directory names
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

    const trackDir = trackDirectoryMap[track] || track
    
    // Read the module JSON data
    const modulesPath = path.join(process.cwd(), 'src', 'data', trackDir, 'modules-descriptions', 'module.json')
    const jsonData = JSON.parse(fs.readFileSync(modulesPath, 'utf-8'))
    
    // Handle different JSON structures
    let modulesData: ModuleData[]
    if (Array.isArray(jsonData)) {
      modulesData = jsonData
    } else if (jsonData.modules && Array.isArray(jsonData.modules)) {
      modulesData = jsonData.modules
    } else {
      console.error('Invalid modules data structure:', typeof jsonData, Object.keys(jsonData))
      return null
    }
    
    // Find the module data by number
    const moduleData = modulesData.find(m => m.id === `${track}-module-${moduleNumber}` || m.id === `module-${moduleNumber}`)
    
    if (!moduleData) {
      return null
    }

    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="space-y-6">
          {/* Module Overview */}
          <div>
            <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-3 flex items-center">
              <span className="mr-2">📚</span>Module Overview
            </h2>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
                {moduleData.title}
              </h3>
              <p className="text-xs text-gray-600 dark:text-gray-300 leading-relaxed">
                {moduleData.subtitle}
              </p>
            </div>
          </div>

          {/* Module Statistics */}
          <div>
            <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-3 flex items-center">
              <span className="mr-2">📊</span>Module Statistics
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 text-center">
                <div className="text-lg font-bold text-blue-600 dark:text-blue-400 mb-1">
                  {moduleData.duration}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Duration</div>
              </div>
              
              <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3 text-center">
                <div className="text-lg font-bold text-green-600 dark:text-green-400 mb-1">
                  {moduleData.lessons}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Lessons</div>
              </div>
              
              <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-3 text-center">
                <div className="text-lg font-bold text-purple-600 dark:text-purple-400 mb-1">
                  {moduleData.labs}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Labs</div>
              </div>
              
              <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-3 text-center">
                <div className="text-lg font-bold text-orange-600 dark:text-orange-400 mb-1">
                  {moduleData.prerequisites.length}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Prerequisites</div>
              </div>
            </div>
          </div>

          {/* Prerequisites */}
          {moduleData.prerequisites.length > 0 && (
            <div>
              <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-3 flex items-center">
                <span className="mr-2">⚡</span>Prerequisites
              </h3>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
                <ul className="space-y-2">
                  {moduleData.prerequisites.map((prereq, index) => (
                    <li key={index} className="flex items-start space-x-2 text-xs">
                      <span className="text-yellow-600 dark:text-yellow-400 mt-0.5 flex-shrink-0">•</span>
                      <span className="text-gray-700 dark:text-gray-300">{prereq}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Key Topics */}
          <div>
            <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-3 flex items-center">
              <span className="mr-2">🎯</span>Key Topics Covered
            </h3>
            <div className="bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg p-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {moduleData.keyTopics.map((topic, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <span className="text-indigo-600 dark:text-indigo-400 flex-shrink-0">▶</span>
                    <span className="text-xs text-gray-700 dark:text-gray-300">{topic}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Skills Gained */}
          <div>
            <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-3 flex items-center">
              <span className="mr-2">🚀</span>Skills You'll Gain
            </h3>
            <div className="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg p-4">
              <ul className="space-y-3">
                {moduleData.skillsGained.map((skill, index) => (
                  <li key={index} className="flex items-start space-x-3">
                    <span className="text-emerald-600 dark:text-emerald-400 mt-0.5 flex-shrink-0">✓</span>
                    <span className="text-xs text-gray-700 dark:text-gray-300 leading-relaxed">{skill}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    )
  } catch (error) {
    console.error('Error loading module data:', error)
    return null
  }
}