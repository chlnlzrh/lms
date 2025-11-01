import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { ModuleCard } from '@/components/content/module-card'
import { Icons } from '@/components/ui/icons'
import { ProgressBar } from '@/components/ui/progress-bar'
import { contentParser } from '@/lib/content-parser'
import { notFound } from 'next/navigation'
import { Suspense } from 'react'

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
  const trackId = resolvedParams.track as 'ai' | 'data-engineering'
  
  if (!['ai', 'data-engineering'].includes(trackId)) {
    notFound()
  }

  const [trackInfo, modules] = await Promise.all([
    contentParser.getTrackInfo(trackId),
    contentParser.getTrackInfo(trackId).then(track => track.modules)
  ])

  if (!trackInfo) {
    notFound()
  }

  const trackIcons = {
    ai: Icons.Bot,
    'data-engineering': Icons.Database
  }

  const trackColors = {
    ai: 'blue',
    'data-engineering': 'green'
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
                  {trackId === 'ai' ? '8.5' : '2.5'}h
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Time Spent</div>
              </div>
              
              <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded">
                <div className={`text-lg font-bold text-${color}-600 mb-1`}>
                  {trackId === 'ai' ? '7' : '3'}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Day Streak</div>
              </div>
            </div>
          </div>
        </div>

        {/* Continue Learning */}
        {totalCompletedLessons > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
              Continue Learning
            </h2>
            
            <div className={`bg-${color}-50 dark:bg-${color}-900/20 border border-${color}-200 dark:border-${color}-800 rounded-lg p-4`}>
              <div className="flex items-center space-x-3">
                <IconComponent className={`w-5 h-5 text-${color}-600`} />
                <div className="flex-1">
                  <h3 className="text-xs font-normal text-gray-900 dark:text-white">
                    {trackId === 'ai' 
                      ? 'Module 1: AI Foundation & Tool Fluency' 
                      : 'Module 1: Database Fundamentals'
                    }
                  </h3>
                  <p className="text-xs text-gray-600 dark:text-gray-300">
                    {trackId === 'ai' 
                      ? 'Last accessed: Prompt Engineering Principles' 
                      : 'Last accessed: ACID Properties'
                    }
                  </p>
                </div>
                <button className={`bg-${color}-600 text-white px-4 py-2 rounded text-xs font-normal hover:bg-${color}-700`}>
                  Continue
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Modules Grid */}
        <div>
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Learning Modules ({modules.length})
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {modules.map((module) => (
              <ModuleCard
                key={module.id}
                module={module}
                progress={mockModuleProgress[module.id as keyof typeof mockModuleProgress]}
              />
            ))}
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