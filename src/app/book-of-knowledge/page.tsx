import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { TrackOverview } from '@/components/content/track-overview'
import { fastContentParser } from '@/lib/fast-content-parser'
import { getTracksByCategory, getCategoryInfo, getTracksMetadata, getTrackStyleInfo } from '@/lib/tracks-metadata'
import { Suspense } from 'react'


async function BookOfKnowledgeContent() {
  // Use pre-built landing page data for maximum performance
  const landingData = await fastContentParser.getAllBookOfKnowledgeTracks()
  
  if (!landingData) {
    return <div>Error loading Book of Knowledge data</div>
  }

  const tracks = landingData.tracks

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Book of Knowledge' }
  ]

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumbs items={breadcrumbItems} />

        {/* Page Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
            {landingData.pageInfo.title}
          </h1>
          <p className="text-xs text-gray-600 dark:text-gray-300">
            {landingData.pageInfo.description}
          </p>
        </div>

        {/* Track Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
            <div className="text-2xl font-bold text-blue-600 mb-1">
              {landingData.overview.totals.lessons}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-300">Total Lessons</div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
            <div className="text-2xl font-bold text-green-600 mb-1">
              {Math.round(landingData.overview.totals.estimatedHours)}h
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-300">Total Hours</div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
            <div className="text-2xl font-bold text-purple-600 mb-1">
              {Math.round(landingData.overview.averages.practicalRatio)}%
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-300">Hands-on Content</div>
          </div>
        </div>

        {/* Knowledge Areas */}
        <div>
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Core Knowledge Areas
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {tracks.map((track) => {
              // Convert landing page track data to TrackOverview format
              const trackOverviewData = {
                id: track.id,
                title: track.title,
                description: track.description,
                totalLessons: track.stats.lessons,
                totalModules: track.stats.modules,
                estimatedDuration: track.stats.duration,
                modules: [] // Not needed for overview
              }
              
              return (
                <TrackOverview
                  key={track.id}
                  track={trackOverviewData}
                  trackIcon={track.icon}
                  trackColor={track.color}
                  basePath="book-of-knowledge"
                />
              )
            })}
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
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <div className="h-96 bg-gray-200 rounded"></div>
            <div className="h-96 bg-gray-200 rounded"></div>
            <div className="h-96 bg-gray-200 rounded"></div>
            <div className="h-96 bg-gray-200 rounded"></div>
            <div className="h-96 bg-gray-200 rounded"></div>
            <div className="h-96 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}

export default function BookOfKnowledgePage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <BookOfKnowledgeContent />
    </Suspense>
  )
}