import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { ModuleCard } from '@/components/content/module-card'
import { Icons } from '@/components/ui/icons'
import { ProgressBar } from '@/components/ui/progress-bar'
import { contentParser } from '@/lib/content-parser'
import { notFound } from 'next/navigation'
import { Suspense } from 'react'
import Link from 'next/link'

interface TrackPageProps {
  params: Promise<{
    track: string
  }>
}

// Mock progress data
const mockModuleProgress = {
  'ai-module-1': { completedLessons: 15, isCompleted: false },
  'ai-module-2': { completedLessons: 0, isCompleted: false },
  'data-engineering-module-1': { completedLessons: 5, isCompleted: false },
}

async function TrackPageContent({ params }: TrackPageProps) {
  const resolvedParams = await params
  const trackId = resolvedParams.track as 'ai' | 'data-engineering' | 'saas'
  
  if (!['ai', 'data-engineering', 'saas'].includes(trackId)) {
    notFound()
  }

  const trackInfo = await contentParser.getTrackInfo(trackId)
  const modules = trackInfo.modules

  if (!trackInfo) {
    notFound()
  }

  const trackIcons = {
    ai: Icons.Bot,
    'data-engineering': Icons.Database,
    'saas': Icons.Cloud
  }

  const trackColors = {
    ai: 'blue',
    'data-engineering': 'green',
    'saas': 'purple'
  }

  const IconComponent = trackIcons[trackId]
  const color = trackColors[trackId]

  // Calculate overall progress
  const totalCompletedLessons = Object.values(mockModuleProgress)
    .filter(progress => progress && modules.some(m => m.id.includes(trackId)))
    .reduce((sum, progress) => sum + progress.completedLessons, 0)
  
  const overallProgress = Math.round((totalCompletedLessons / trackInfo.totalLessons) * 100)

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Talent Development', href: '/talent-development' },
    { label: trackInfo.title }
  ]

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumbs items={breadcrumbItems} />

        {/* Track Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-start space-x-4">
            <IconComponent className={`w-8 h-8 text-${color}-600 mt-1 flex-shrink-0`} />
            
            <div className="flex-1">
              <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                {trackInfo.title}
              </h1>
              <p className="text-xs text-gray-600 dark:text-gray-300 mb-4">
                {trackInfo.description}
              </p>
              
              {/* Track Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className={`text-2xl font-bold text-${color}-600 mb-1`}>
                    {trackInfo.totalModules}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Modules</div>
                </div>
                
                <div className="text-center">
                  <div className={`text-2xl font-bold text-${color}-600 mb-1`}>
                    {trackInfo.totalLessons}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Lessons</div>
                </div>
                
                <div className="text-center">
                  <div className={`text-2xl font-bold text-${color}-600 mb-1`}>
                    {trackInfo.estimatedDuration}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Duration</div>
                </div>
                
                <div className="text-center">
                  <div className={`text-2xl font-bold text-${color}-600 mb-1`}>
                    {overallProgress}%
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Complete</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Overall Progress */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Your Progress
          </h2>
          
          <div className="space-y-3">
            <div className="flex justify-between text-xs">
              <span className="text-gray-600 dark:text-gray-300">Overall Track Progress</span>
              <span className="text-gray-900 dark:text-white">
                {totalCompletedLessons} of {trackInfo.totalLessons} lessons completed
              </span>
            </div>
            <ProgressBar 
              value={totalCompletedLessons} 
              max={trackInfo.totalLessons} 
              showPercentage 
            />
            
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <div className={`text-lg font-bold text-${color}-600 mb-1`}>
                  {Object.values(mockModuleProgress).filter(p => p?.isCompleted).length}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Modules Completed</div>
              </div>
              
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <div className={`text-lg font-bold text-${color}-600 mb-1`}>
                  {trackId === 'ai' ? '8.5' : trackId === 'saas' ? '12.3' : '2.5'}h
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Time Spent</div>
              </div>
              
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <div className={`text-lg font-bold text-${color}-600 mb-1`}>
                  {trackId === 'ai' ? '7' : trackId === 'saas' ? '5' : '3'}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Day Streak</div>
              </div>
            </div>
          </div>
        </div>

        {/* Getting Started / Continue Learning */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-sm font-bold text-gray-900 dark:text-white mb-4">
            {totalCompletedLessons > 0 ? 'Continue Your Learning Journey' : 'Start Your Learning Journey'}
          </h2>
          
          <div className={`bg-${color}-50 dark:bg-${color}-900/20 border border-${color}-200 dark:border-${color}-800 rounded-lg p-4`}>
            <div className="flex items-center space-x-4">
              <IconComponent className={`w-8 h-8 text-${color}-600`} />
              <div className="flex-1">
                <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-1">
                  {totalCompletedLessons > 0 
                    ? `Continue with ${trackId === 'ai' ? 'AI Foundation & Tool Fluency' : trackId === 'saas' ? 'SaaS Architecture & System Design' : 'Database Fundamentals'}`
                    : `Begin with ${trackId === 'ai' ? 'AI Foundation & Tool Fluency' : trackId === 'saas' ? 'SaaS Architecture & System Design' : 'Database Fundamentals'}`
                  }
                </h3>
                <p className="text-xs text-gray-600 dark:text-gray-300 mb-3">
                  {totalCompletedLessons > 0 
                    ? `${totalCompletedLessons} lessons completed • Keep up the momentum!`
                    : `Start with Module 1 and build your ${trackId === 'ai' ? 'AI expertise' : trackId === 'saas' ? 'SaaS development skills' : 'data engineering skills'} step by step`
                  }
                </p>
                <div className="flex items-center space-x-3">
                  <Link
                    href={`/talent-development/${trackId}/module-1`}
                    className={`bg-${color}-600 text-white px-4 py-2 rounded text-xs font-normal hover:bg-${color}-700 transition-colors`}
                  >
                    {totalCompletedLessons > 0 ? 'Continue Learning' : 'Start Learning'}
                  </Link>
                  <Link
                    href="/talent-development"
                    className="text-xs text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
                  >
                    ← Back to All Tracks
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Track Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <h2 className="text-sm font-bold text-gray-900 dark:text-white mb-4">
              What You'll Learn
            </h2>
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {trackId === 'ai' ? (
                  <>
                    <div className="flex items-start space-x-3">
                      <Icons.Bot className="w-5 h-5 text-blue-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">AI Tool Mastery</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Claude Code, GPT-4, and advanced AI development tools</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <Icons.Settings className="w-5 h-5 text-blue-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">Prompt Engineering</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Advanced techniques for AI interaction and automation</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <Icons.Shield className="w-5 h-5 text-blue-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">AI Safety & Ethics</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Responsible AI development and governance</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <Icons.BookOpen className="w-5 h-5 text-blue-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">Agent Architecture</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Build sophisticated AI agents and platforms</p>
                      </div>
                    </div>
                  </>
                ) : trackId === 'saas' ? (
                  <>
                    <div className="flex items-start space-x-3">
                      <Icons.Cloud className="w-5 h-5 text-purple-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">Multi-Tenant Architecture</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Scalable SaaS architecture patterns and isolation strategies</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <Icons.Settings className="w-5 h-5 text-purple-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">Modern Stack</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">TypeScript, React, Next.js, Node.js, and PostgreSQL</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <Icons.Shield className="w-5 h-5 text-purple-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">Security & Compliance</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Authentication, authorization, and enterprise compliance</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <Icons.BookOpen className="w-5 h-5 text-purple-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">Platform Engineering</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">CI/CD, observability, and production-ready systems</p>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="flex items-start space-x-3">
                      <Icons.Database className="w-5 h-5 text-green-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">Database Design</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Relational design, normalization, and optimization</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <Icons.Settings className="w-5 h-5 text-green-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">ETL & Pipelines</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Data extraction, transformation, and loading processes</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <Icons.Cloud className="w-5 h-5 text-green-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">Snowflake Platform</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Cloud data warehousing and analytics</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <Icons.Shield className="w-5 h-5 text-green-600 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">Data Governance</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">Quality, security, and compliance best practices</p>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
          
          <div>
            <h2 className="text-sm font-bold text-gray-900 dark:text-white mb-4">
              Track Stats
            </h2>
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="space-y-4">
                <div className="text-center">
                  <div className={`text-2xl font-bold text-${color}-600 mb-1`}>
                    {trackInfo.totalLessons}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Total Lessons</div>
                </div>
                <div className="text-center">
                  <div className={`text-2xl font-bold text-${color}-600 mb-1`}>
                    {modules.length}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Learning Modules</div>
                </div>
                <div className="text-center">
                  <div className={`text-2xl font-bold text-${color}-600 mb-1`}>
                    {trackInfo.estimatedDuration}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Est. Duration</div>
                </div>
                <div className="text-center">
                  <div className={`text-2xl font-bold text-${color}-600 mb-1`}>
                    {overallProgress}%
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Your Progress</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Modules Grid - Enhanced for Track View */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              Learning Modules ({modules.length})
            </h2>
            <div className="text-xs text-gray-500">
              Click any module to start learning
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {modules.map((module, index) => {
              const moduleProgress = mockModuleProgress[module.id as keyof typeof mockModuleProgress]
              const isCompleted = moduleProgress?.isCompleted || false
              const completedLessons = moduleProgress?.completedLessons || 0
              
              return (
                <Link
                  key={module.id}
                  href={`/talent-development/${trackId}/module-${module.moduleNumber}`}
                  className="block bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:border-gray-300 dark:hover:border-gray-600 transition-all duration-200 hover:shadow-md"
                >
                  <div className="flex items-start space-x-4">
                    <div className={`w-12 h-12 rounded-lg bg-${color}-100 dark:bg-${color}-900 flex items-center justify-center flex-shrink-0`}>
                      <span className={`text-lg font-bold text-${color}-600`}>
                        {index + 1}
                      </span>
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-2">
                        <h3 className="text-sm font-bold text-gray-900 dark:text-white">
                          {module.title}
                        </h3>
                        {isCompleted && (
                          <Icons.Circle className="w-4 h-4 text-green-600 fill-current" />
                        )}
                      </div>
                      
                      <p className="text-xs text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
                        {module.description}
                      </p>
                      
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span className="flex items-center space-x-1">
                            <Icons.BookOpen className="w-3 h-3" />
                            <span>{module.lessonCount || 'Multiple'} lessons</span>
                          </span>
                          <span className="flex items-center space-x-1">
                            <Icons.Clock className="w-3 h-3" />
                            <span>{module.duration || '2-4 weeks'}</span>
                          </span>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {completedLessons > 0 && (
                            <span className="text-xs px-2 py-1 rounded bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300">
                              {completedLessons} completed
                            </span>
                          )}
                          <Icons.ChevronRight className="w-4 h-4 text-gray-400" />
                        </div>
                      </div>
                    </div>
                  </div>
                </Link>
              )
            })}
          </div>
        </div>

        {/* Learning Path */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Recommended Learning Path
          </h2>
          
          <div className="space-y-3">
            {modules.slice(0, 5).map((module, index) => (
              <div key={module.id} className="flex items-center space-x-3">
                <div className={`w-8 h-8 rounded-full bg-${color}-100 dark:bg-${color}-900 flex items-center justify-center`}>
                  <span className={`text-xs font-bold text-${color}-600`}>
                    {index + 1}
                  </span>
                </div>
                
                <div className="flex-1">
                  <div className="text-xs font-normal text-gray-900 dark:text-white">
                    {module.title}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">
                    {module.lessonCount} lessons • {module.duration}
                  </div>
                </div>
                
                <div className="text-xs text-gray-500">
                  {mockModuleProgress[module.id as keyof typeof mockModuleProgress]?.isCompleted 
                    ? '✓ Complete' 
                    : mockModuleProgress[module.id as keyof typeof mockModuleProgress]?.completedLessons > 0
                      ? 'In Progress'
                      : 'Not Started'
                  }
                </div>
              </div>
            ))}
          </div>
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
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-64 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    </MainLayout>
  )
}

export default function TrackPage({ params }: TrackPageProps) {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <TrackPageContent params={params} />
    </Suspense>
  )
}