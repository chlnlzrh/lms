import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { TrackOverview } from '@/components/content/track-overview'
import { contentParser } from '@/lib/content-parser'
import { Suspense } from 'react'

// Mock progress data - in real app this would come from user state
const mockProgress = {
  ai: {
    completedLessons: 15,
    completedModules: 1,
    timeSpent: 8.5
  },
  'data-engineering': {
    completedLessons: 5,
    completedModules: 0,
    timeSpent: 2.5
  }
}

async function TalentDevelopmentContent() {
  const [aiTrack, deTrack] = await Promise.all([
    contentParser.getTrackInfo('ai'),
    contentParser.getTrackInfo('data-engineering')
  ])
  const tracks = [aiTrack, deTrack]

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Talent Development' }
  ]

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumbs items={breadcrumbItems} />

        {/* Page Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
            Talent Development
          </h1>
          <p className="text-xs text-gray-600 dark:text-gray-300">
            Choose your learning track and advance your professional skills through comprehensive, hands-on training programs.
          </p>
        </div>

        {/* Track Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
            <div className="text-2xl font-bold text-blue-600 mb-1">
              {tracks.reduce((total, track) => total + track.totalLessons, 0)}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-300">Total Lessons</div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
            <div className="text-2xl font-bold text-green-600 mb-1">
              {tracks.reduce((total, track) => total + track.totalModules, 0)}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-300">Total Modules</div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
            <div className="text-2xl font-bold text-purple-600 mb-1">6</div>
            <div className="text-xs text-gray-600 dark:text-gray-300">Learning Tracks</div>
            <div className="text-xs text-orange-600 mt-1">2 Active, 4 Coming Soon</div>
          </div>
        </div>

        {/* Active Tracks */}
        <div>
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Active Learning Tracks
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {tracks.map((track) => (
              <TrackOverview
                key={track.id}
                track={track}
                progress={mockProgress[track.id]}
              />
            ))}
          </div>
        </div>

        {/* Coming Soon Tracks */}
        <div>
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Coming Soon
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { name: 'Integration Track', icon: '🔗', description: 'API design, Workato, enterprise patterns' },
              { name: 'SaaS App Build Track', icon: '⚙️', description: 'Multi-tenant architecture, cloud services' },
              { name: 'Salesforce Track', icon: '☁️', description: 'Apex, Lightning, platform development' },
              { name: 'MDM & Data Governance Track', icon: '🛡️', description: 'Data quality, governance, compliance' }
            ].map((track) => (
              <div
                key={track.name}
                className="bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center opacity-75"
              >
                <div className="text-3xl mb-3">{track.icon}</div>
                <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-2">
                  {track.name}
                </h3>
                <p className="text-xs text-gray-600 dark:text-gray-300">
                  {track.description}
                </p>
                <div className="mt-4">
                  <span className="text-xs bg-orange-100 text-orange-600 px-3 py-1 rounded">
                    Coming Soon
                  </span>
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
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="h-32 bg-gray-200 rounded mb-6"></div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-96 bg-gray-200 rounded"></div>
            <div className="h-96 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}

export default function TalentDevelopmentPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <TalentDevelopmentContent />
    </Suspense>
  )
}