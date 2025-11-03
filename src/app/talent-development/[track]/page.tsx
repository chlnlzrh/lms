import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { ModuleCard } from '@/components/content/module-card'
import { Icons } from '@/components/ui/icons'
import { fastContentParser } from '@/lib/fast-content-parser'
import { notFound } from 'next/navigation'
import { Suspense } from 'react'
import Link from 'next/link'

interface TrackPageProps {
  params: Promise<{
    track: string
  }>
}


async function TrackPageContent({ params }: TrackPageProps) {
  const resolvedParams = await params
  const trackId = resolvedParams.track
  
  // Validate track exists by trying to get track info (using fast parser)
  let trackInfo
  try {
    trackInfo = await fastContentParser.getTrackInfo(trackId)
    if (!trackInfo || trackInfo.totalModules === 0) {
      notFound()
    }
  } catch {
    notFound()
  }

  const modules = trackInfo.modules

  // Use a simple mapping for track icons and colors (avoiding server-side import issues)
  const trackIcons: { [key: string]: any } = {
    ai: Icons.Bot,
    de: Icons.Database,
    saas: Icons.Cloud,
    sfdc: Icons.Zap,
    snowflake_tune: Icons.Snowflake,
    workato: Icons.Link,
    ba: Icons.BarChart3,
    data_engineer: Icons.Database,
    data_gov: Icons.Shield,
    devops_engineer: Icons.Settings,
    finance: Icons.DollarSign,
    hr: Icons.Users,
    mdm: Icons.Archive,
    pm: Icons.Briefcase,
    qa: Icons.CheckCircle,
    rpa: Icons.Bot,
    sales: Icons.TrendingUp,
    sfdc_engineer: Icons.Zap,
    ta: Icons.FileSearch,
    viz_engineer: Icons.PieChart,
    workato_engineer: Icons.Link
  }

  const trackColors: { [key: string]: string } = {
    ai: 'blue',
    de: 'green',
    saas: 'purple',
    sfdc: 'blue',
    snowflake_tune: 'cyan',
    workato: 'teal',
    ba: 'orange',
    data_engineer: 'green',
    data_gov: 'emerald',
    devops_engineer: 'slate',
    finance: 'green',
    hr: 'pink',
    mdm: 'indigo',
    pm: 'yellow',
    qa: 'green',
    rpa: 'orange',
    sales: 'red',
    sfdc_engineer: 'blue',
    ta: 'gray',
    viz_engineer: 'purple',
    workato_engineer: 'teal'
  }

  const IconComponent = trackIcons[trackId] || Icons.BookOpen
  const color = trackColors[trackId] || 'blue'

  // Use pre-computed statistics from fast parser
  const totalCompletedLessons = 0 // TODO: Get from user progress
  const overallProgress = 0 // TODO: Calculate from user progress
  const estimatedHours = trackInfo.stats?.estimatedHours || 0
  const practicalRatio = trackInfo.stats?.practicalRatio || 0

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
                    {Math.round(estimatedHours)}h
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Est. Hours</div>
                </div>
                
                <div className="text-center">
                  <div className={`text-2xl font-bold text-${color}-600 mb-1`}>
                    {Math.round(practicalRatio)}%
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Hands-on</div>
                </div>
              </div>
            </div>
          </div>
        </div>


        {/* Getting Started */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-sm font-bold text-gray-900 dark:text-white mb-4">
            Start Your Learning Journey
          </h2>
          
          <div className={`bg-${color}-50 dark:bg-${color}-900/20 border border-${color}-200 dark:border-${color}-800 rounded-lg p-4`}>
            <div className="flex items-center space-x-4">
              <IconComponent className={`w-8 h-8 text-${color}-600`} />
              <div className="flex-1">
                <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-1">
                  Begin with {modules.length > 0 ? modules[0].title : 'Module 1'}
                </h3>
                <p className="text-xs text-gray-600 dark:text-gray-300 mb-3">
                  Start your {trackInfo.title.toLowerCase()} journey with our structured learning path
                </p>
                <div className="flex items-center space-x-3">
                  <Link
                    href={`/talent-development/${trackId}/module-${modules.length > 0 ? modules[0].moduleNumber || 1 : 1}`}
                    className={`bg-${color}-600 text-white px-4 py-2 rounded text-xs font-normal hover:bg-${color}-700 transition-colors`}
                  >
                    Start Learning
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
                {modules.slice(0, 4).map((module, index) => {
                  const icons = [Icons.BookOpen, Icons.Settings, Icons.Shield, Icons.Database, Icons.Cloud, Icons.Bot]
                  const IconComponent = icons[index % icons.length]
                  
                  return (
                    <div key={module.id} className="flex items-start space-x-3">
                      <IconComponent className={`w-5 h-5 text-${color}-600 mt-0.5`} />
                      <div>
                        <h4 className="text-xs font-bold text-gray-900 dark:text-white mb-1">{module.title}</h4>
                        <p className="text-xs text-gray-600 dark:text-gray-300">{module.description}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
              
              {modules.length === 0 && (
                <div className="text-center py-8">
                  <Icons.BookOpen className={`w-12 h-12 text-${color}-600 mx-auto mb-3`} />
                  <p className="text-xs text-gray-600 dark:text-gray-300">
                    Content is being developed for this track. Check back soon!
                  </p>
                </div>
              )}
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
                    Active
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">Status</div>
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
            {modules.map((module, index) => (
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
                        <Icons.ChevronRight className="w-4 h-4 text-gray-400" />
                      </div>
                    </div>
                  </div>
                </div>
              </Link>
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
                  Available
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