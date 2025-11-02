import { notFound } from 'next/navigation'
import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { Icons } from '@/components/ui/icons'
import { ProgressBar } from '@/components/ui/progress-bar'
import { contentParser } from '@/lib/content-parser'
import { progressTracker } from '@/lib/progress-tracker'
import { ModuleDetailsSection } from '@/components/module/module-details-section'
import { Suspense } from 'react'
import Link from 'next/link'

interface ModulePageProps {
  params: Promise<{
    track: string
    module: string
  }>
}



async function ModulePageContent({ params }: ModulePageProps) {
  const resolvedParams = await params
  const { track, module } = resolvedParams
  
  if (!['ai', 'data-engineering', 'saas'].includes(track)) {
    notFound()
  }

  // Get track info and modules
  const trackInfo = await contentParser.getTrackInfo(track as 'ai' | 'data-engineering' | 'saas')
  const modules = trackInfo.modules
  // Extract module number from module parameter (e.g., "module-1" -> 1)
  const moduleNumber = parseInt(module.replace('module-', ''))
  
  // Find the specific module
  const selectedModule = modules.find(m => m.moduleNumber === moduleNumber)
  if (!selectedModule) {
    notFound()
  }

  // Get all lessons for this track and filter by module
  const allLessons = await contentParser.getAllLessons(track as 'ai' | 'data-engineering' | 'saas')
  const moduleLessons = allLessons.filter(lesson => lesson.moduleNumber === moduleNumber)

  // Get user progress
  const userProgress = progressTracker.getUserProgress()
  
  // Calculate module progress
  const completedLessons = moduleLessons.filter(lesson => 
    userProgress.lessons[lesson.id]?.isCompleted
  ).length
  const totalLessons = moduleLessons.length
  const progressPercentage = totalLessons > 0 ? Math.round((completedLessons / totalLessons) * 100) : 0

  // Calculate total time spent in this module
  const totalTimeSpent = moduleLessons.reduce((total, lesson) => {
    const lessonProgress = userProgress.lessons[lesson.id]
    return total + (lessonProgress?.timeSpent || 0)
  }, 0)

  const trackColors = {
    ai: 'blue',
    'data-engineering': 'green',
    'saas': 'purple'
  }

  const trackIcons = {
    ai: Icons.Bot,
    'data-engineering': Icons.Database,
    'saas': Icons.Cloud
  }

  const color = trackColors[track as keyof typeof trackColors]
  const IconComponent = trackIcons[track as keyof typeof trackIcons]

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Talent Development', href: '/talent-development' },
    { 
      label: track === 'ai' ? 'AI Training' : track === 'saas' ? 'SaaS Development' : 'Data Engineering', 
      href: `/talent-development/${track}` 
    },
    { label: selectedModule.title }
  ]

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumbs items={breadcrumbItems} />

        {/* Module Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-start space-x-4">
            <IconComponent className={`w-8 h-8 text-${color}-600 mt-1 flex-shrink-0`} />
            
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-2">
                <span className={`text-xs px-2 py-1 rounded bg-${color}-100 text-${color}-700 dark:bg-${color}-900 dark:text-${color}-300`}>
                  Module {moduleNumber}
                </span>
                <span className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                  {totalLessons} lessons
                </span>
              </div>
              
              <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                {selectedModule.title}
              </h1>
              
              <p className="text-xs text-gray-600 dark:text-gray-300 mb-4">
                {selectedModule.description}
              </p>

              {/* Module Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                  <div className={`text-lg font-bold text-${color}-600 mb-1`}>
                    {completedLessons}/{totalLessons}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Lessons</div>
                </div>
                
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                  <div className={`text-lg font-bold text-${color}-600 mb-1`}>
                    {progressPercentage}%
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Complete</div>
                </div>
                
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                  <div className={`text-lg font-bold text-${color}-600 mb-1`}>
                    {Math.round(totalTimeSpent / 60 * 10) / 10}h
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Time Spent</div>
                </div>
                
                <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                  <div className={`text-lg font-bold text-${color}-600 mb-1`}>
                    {selectedModule.duration || 'N/A'}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Duration</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Progress Overview */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Module Progress
          </h2>
          
          <div className="space-y-3">
            <div className="flex justify-between text-xs">
              <span className="text-gray-600 dark:text-gray-300">Overall Progress</span>
              <span className="text-gray-900 dark:text-white">
                {completedLessons} of {totalLessons} lessons completed
              </span>
            </div>
            <ProgressBar 
              value={completedLessons} 
              max={totalLessons} 
              showPercentage 
            />
          </div>
        </div>

        {/* Enhanced Module Information */}
        <ModuleDetailsSection moduleNumber={moduleNumber} color={color} />

        {/* Learning Objectives */}
        {selectedModule.learningObjectives && selectedModule.learningObjectives.length > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
              Learning Objectives
            </h2>
            
            <ul className="space-y-2">
              {selectedModule.learningObjectives.map((objective, index) => (
                <li key={index} className="flex items-start space-x-2 text-xs">
                  <span className={`text-${color}-500 mt-0.5`}>•</span>
                  <span className="text-gray-600 dark:text-gray-300">{objective}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Lessons List */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Lessons ({moduleLessons.length})
          </h2>
          
          {moduleLessons.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {moduleLessons.map((lesson, index) => {
                const lessonProgress = userProgress.lessons[lesson.id]
                const isCompleted = lessonProgress?.isCompleted || false
                const isBookmarked = lessonProgress?.bookmarked || false
                const timeSpent = lessonProgress?.timeSpent || 0
                
                return (
                  <Link
                    key={lesson.id}
                    href={`/talent-development/${track}/lesson/${lesson.slug}`}
                    className="block p-4 border border-gray-100 dark:border-gray-700 rounded hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0">
                        <div className={`w-8 h-8 rounded-full bg-${color}-100 dark:bg-${color}-900 flex items-center justify-center`}>
                          <span className={`text-xs font-bold text-${color}-600`}>
                            {index + 1}
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-xs font-normal text-gray-900 dark:text-white truncate">
                            {lesson.frontmatter.title}
                          </h3>
                          
                          <div className="flex items-center space-x-1">
                            {isCompleted && (
                              <Icons.Circle className="w-3 h-3 text-green-600 fill-current" />
                            )}
                            {isBookmarked && (
                              <Icons.Bookmark className="w-3 h-3 text-yellow-600 fill-current" />
                            )}
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                          <span className="flex items-center space-x-1">
                            <Icons.Clock className="w-3 h-3" />
                            <span>{lesson.estimatedReadTime} min read</span>
                          </span>
                          
                          {timeSpent > 0 && (
                            <span className="flex items-center space-x-1">
                              <Icons.Timer className="w-3 h-3" />
                              <span>{timeSpent} min spent</span>
                            </span>
                          )}
                          
                          {lesson.frontmatter.complexity && (
                            <span className="px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
                              {lesson.frontmatter.complexity === 'F' ? 'Foundation' : 
                               lesson.frontmatter.complexity === 'I' ? 'Intermediate' : 'Advanced'}
                            </span>
                          )}
                        </div>
                      </div>
                      
                      <div className="flex-shrink-0">
                        <Icons.ChevronRight className="w-4 h-4 text-gray-400" />
                      </div>
                    </div>
                  </Link>
                )
              })}
            </div>
          ) : (
            <div className="text-center py-8">
              <Icons.BookOpen className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p className="text-xs text-gray-500 dark:text-gray-400">
                No lessons found for this module.
              </p>
            </div>
          )}
        </div>

        {/* Module Navigation */}
        <div className="flex items-center justify-between">
          {moduleNumber > 1 ? (
            <Link
              href={`/talent-development/${track}/module-${moduleNumber - 1}`}
              className="flex items-center space-x-2 text-xs text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              <Icons.ChevronLeft className="w-4 h-4" />
              <span>Previous Module</span>
            </Link>
          ) : (
            <div></div>
          )}

          {moduleNumber < modules.length ? (
            <Link
              href={`/talent-development/${track}/module-${moduleNumber + 1}`}
              className="flex items-center space-x-2 text-xs text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              <span>Next Module</span>
              <Icons.ChevronRight className="w-4 h-4" />
            </Link>
          ) : (
            <div></div>
          )}
        </div>
      </div>
    </MainLayout>
  )
}

function LoadingFallback() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="h-48 bg-gray-200 rounded mb-6"></div>
          <div className="h-32 bg-gray-200 rounded mb-6"></div>
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-16 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    </MainLayout>
  )
}


export default function ModulePage({ params }: ModulePageProps) {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <ModulePageContent params={params} />
    </Suspense>
  )
}