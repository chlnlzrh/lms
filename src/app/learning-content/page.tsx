import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { Icons } from '@/components/ui/icons'
import { fastContentParser } from '@/lib/fast-content-parser'
import { Suspense } from 'react'
import Link from 'next/link'

async function LearningContentContent() {
  // Load both Book of Knowledge and Learning Path data
  const [bookOfKnowledgeData, learningPathData] = await Promise.all([
    fastContentParser.getAllBookOfKnowledgeTracks(),
    fastContentParser.getAllLearningPathTracks()
  ])

  if (!bookOfKnowledgeData || !learningPathData) {
    return <div>Error loading Learning Content data</div>
  }

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Learning Content' }
  ]

  // Calculate combined statistics
  const totalLessons = bookOfKnowledgeData.overview.totals.lessons + learningPathData.overview.totals.lessons
  const totalHours = Math.round(bookOfKnowledgeData.overview.totals.estimatedHours + learningPathData.overview.totals.estimatedHours)
  const totalTracks = bookOfKnowledgeData.tracks.length + learningPathData.tracks.length

  // Icon mapping for consistent display
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
    marketing: Icons.Megaphone,
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

  // Color mapping for consistent styling
  const getColorClasses = (color: string) => {
    const colorMap: { [key: string]: { bg: string, icon: string, button: string, hover: string } } = {
      blue: { bg: 'bg-blue-100 dark:bg-blue-900', icon: 'text-blue-600', button: 'bg-blue-600', hover: 'hover:bg-blue-700' },
      green: { bg: 'bg-green-100 dark:bg-green-900', icon: 'text-green-600', button: 'bg-green-600', hover: 'hover:bg-green-700' },
      purple: { bg: 'bg-purple-100 dark:bg-purple-900', icon: 'text-purple-600', button: 'bg-purple-600', hover: 'hover:bg-purple-700' },
      orange: { bg: 'bg-orange-100 dark:bg-orange-900', icon: 'text-orange-600', button: 'bg-orange-600', hover: 'hover:bg-orange-700' },
      red: { bg: 'bg-red-100 dark:bg-red-900', icon: 'text-red-600', button: 'bg-red-600', hover: 'hover:bg-red-700' },
      yellow: { bg: 'bg-yellow-100 dark:bg-yellow-900', icon: 'text-yellow-600', button: 'bg-yellow-600', hover: 'hover:bg-yellow-700' },
      indigo: { bg: 'bg-indigo-100 dark:bg-indigo-900', icon: 'text-indigo-600', button: 'bg-indigo-600', hover: 'hover:bg-indigo-700' },
      pink: { bg: 'bg-pink-100 dark:bg-pink-900', icon: 'text-pink-600', button: 'bg-pink-600', hover: 'hover:bg-pink-700' },
      teal: { bg: 'bg-teal-100 dark:bg-teal-900', icon: 'text-teal-600', button: 'bg-teal-600', hover: 'hover:bg-teal-700' },
      cyan: { bg: 'bg-cyan-100 dark:bg-cyan-900', icon: 'text-cyan-600', button: 'bg-cyan-600', hover: 'hover:bg-cyan-700' },
      gray: { bg: 'bg-gray-100 dark:bg-gray-900', icon: 'text-gray-600', button: 'bg-gray-600', hover: 'hover:bg-gray-700' }
    }
    return colorMap[color] || colorMap.blue
  }

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumbs items={breadcrumbItems} />

        {/* Page Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
            Learning Content
          </h1>
          <p className="text-xs text-gray-600 dark:text-gray-300">
            Explore comprehensive learning materials and structured career paths to advance your professional skills through hands-on training programs.
          </p>
        </div>

        {/* Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
            <div className="text-2xl font-bold text-blue-600 mb-1">
              {totalLessons}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-300">Total Lessons</div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
            <div className="text-2xl font-bold text-green-600 mb-1">
              {totalHours}h
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-300">Total Hours</div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
            <div className="text-2xl font-bold text-purple-600 mb-1">
              {totalTracks}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-300">Learning Tracks</div>
          </div>
        </div>

        {/* Book of Knowledge Table */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-3">
              <Icons.BookOpen className="w-5 h-5 text-blue-600" />
              <h2 className="text-xs font-bold text-gray-900 dark:text-white">
                Book of Knowledge ({bookOfKnowledgeData.tracks.length})
              </h2>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-300 mt-2">
              Core knowledge areas and fundamental concepts
            </p>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-bold text-gray-900 dark:text-white">Track</th>
                  <th className="px-6 py-3 text-left text-xs font-bold text-gray-900 dark:text-white">Description</th>
                  <th className="px-6 py-3 text-center text-xs font-bold text-gray-900 dark:text-white">Lessons</th>
                  <th className="px-6 py-3 text-center text-xs font-bold text-gray-900 dark:text-white">Duration</th>
                  <th className="px-6 py-3 text-center text-xs font-bold text-gray-900 dark:text-white">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {bookOfKnowledgeData.tracks.map((track) => {
                  const IconComponent = trackIcons[track.id] || Icons.BookOpen
                  const colors = getColorClasses(track.color)
                  return (
                    <tr key={track.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                      <td className="px-6 py-4">
                        <div className="flex items-center space-x-3">
                          <div className={`w-8 h-8 rounded-lg ${colors.bg} flex items-center justify-center`}>
                            <IconComponent className={`w-4 h-4 ${colors.icon}`} />
                          </div>
                          <div>
                            <div className="text-xs font-bold text-gray-900 dark:text-white">{track.title}</div>
                          </div>
                        </div>
                      </td>
                    <td className="px-6 py-4">
                      <div className="text-xs text-gray-600 dark:text-gray-300 line-clamp-2">
                        {track.description}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <div className="text-xs text-gray-900 dark:text-white">{track.stats.lessons}</div>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <div className="text-xs text-gray-900 dark:text-white">{track.stats.duration}</div>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <Link
                        href={`/book-of-knowledge/${track.id}`}
                        className={`${colors.button} text-white px-3 py-1 rounded text-xs font-normal ${colors.hover} transition-colors`}
                      >
                        Explore
                      </Link>
                    </td>
                  </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Learning Path Table */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-3">
              <Icons.GraduationCap className="w-5 h-5 text-green-600" />
              <h2 className="text-xs font-bold text-gray-900 dark:text-white">
                Learning Path ({learningPathData.tracks.length})
              </h2>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-300 mt-2">
              Structured career paths and professional development tracks
            </p>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-bold text-gray-900 dark:text-white">Track</th>
                  <th className="px-6 py-3 text-left text-xs font-bold text-gray-900 dark:text-white">Description</th>
                  <th className="px-6 py-3 text-center text-xs font-bold text-gray-900 dark:text-white">Lessons</th>
                  <th className="px-6 py-3 text-center text-xs font-bold text-gray-900 dark:text-white">Duration</th>
                  <th className="px-6 py-3 text-center text-xs font-bold text-gray-900 dark:text-white">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {learningPathData.tracks.map((track) => {
                  const IconComponent = trackIcons[track.id] || Icons.GraduationCap
                  const colors = getColorClasses(track.color)
                  return (
                    <tr key={track.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                      <td className="px-6 py-4">
                        <div className="flex items-center space-x-3">
                          <div className={`w-8 h-8 rounded-lg ${colors.bg} flex items-center justify-center`}>
                            <IconComponent className={`w-4 h-4 ${colors.icon}`} />
                          </div>
                          <div>
                            <div className="text-xs font-bold text-gray-900 dark:text-white">{track.title}</div>
                          </div>
                        </div>
                      </td>
                    <td className="px-6 py-4">
                      <div className="text-xs text-gray-600 dark:text-gray-300 line-clamp-2">
                        {track.description}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <div className="text-xs text-gray-900 dark:text-white">{track.stats.lessons}</div>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <div className="text-xs text-gray-900 dark:text-white">{track.stats.duration}</div>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <Link
                        href={`/learning-path/${track.id}`}
                        className={`${colors.button} text-white px-3 py-1 rounded text-xs font-normal ${colors.hover} transition-colors`}
                      >
                        Explore
                      </Link>
                    </td>
                  </tr>
                  )
                })}
              </tbody>
            </table>
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
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="h-32 bg-gray-200 rounded mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="h-24 bg-gray-200 rounded"></div>
            <div className="h-24 bg-gray-200 rounded"></div>
            <div className="h-24 bg-gray-200 rounded"></div>
          </div>
          <div className="h-96 bg-gray-200 rounded mb-6"></div>
          <div className="h-96 bg-gray-200 rounded"></div>
        </div>
      </div>
    </MainLayout>
  )
}

export default function LearningContentPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <LearningContentContent />
    </Suspense>
  )
}