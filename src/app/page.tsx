import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { ProgressBar } from '@/components/ui/progress-bar'
import { Icons } from '@/components/ui/icons'

export default function HomePage() {
  const breadcrumbItems = [
    { label: 'Dashboard' }
  ]

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumbs items={breadcrumbItems} />

        {/* Welcome Section */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
            Welcome to the Learning Management System
          </h1>
          <p className="text-xs text-gray-600 dark:text-gray-300">
            Continue your learning journey across AI Training, Data Engineering, and SaaS Development tracks.
          </p>
        </div>

        {/* Continue Learning */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Continue Learning
          </h2>
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <Icons.Bot className="w-5 h-5 text-blue-600" />
              <div className="flex-1">
                <h3 className="text-xs font-normal text-gray-900 dark:text-white">
                  AI Training Track - Module 1: AI Foundation & Tool Fluency
                </h3>
                <p className="text-xs text-gray-600 dark:text-gray-300">
                  Last accessed: Prompt Engineering Principles
                </p>
              </div>
              <button className="bg-blue-600 text-white px-4 py-2 rounded text-xs font-normal hover:bg-blue-700">
                Continue
              </button>
            </div>
          </div>
        </div>

        {/* Progress Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* AI Training Track Progress */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Icons.Bot className="w-5 h-5 text-blue-600" />
              <h3 className="text-xs font-bold text-gray-900 dark:text-white">
                AI Training Track
              </h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-gray-600 dark:text-gray-300">Progress</span>
                <span className="text-gray-900 dark:text-white">15 of 234 lessons</span>
              </div>
              <ProgressBar value={15} max={234} showPercentage />
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-gray-600 dark:text-gray-300">Modules</span>
                  <div className="text-gray-900 dark:text-white font-normal">1 of 6</div>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-300">Time Spent</span>
                  <div className="text-gray-900 dark:text-white font-normal">8.5 hours</div>
                </div>
              </div>
            </div>
          </div>

          {/* Data Engineering Track Progress */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Icons.Database className="w-5 h-5 text-green-600" />
              <h3 className="text-xs font-bold text-gray-900 dark:text-white">
                Data Engineering Track
              </h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-gray-600 dark:text-gray-300">Progress</span>
                <span className="text-gray-900 dark:text-white">5 of 300+ lessons</span>
              </div>
              <ProgressBar value={5} max={300} showPercentage />
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-gray-600 dark:text-gray-300">Modules</span>
                  <div className="text-gray-900 dark:text-white font-normal">0 of 20</div>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-300">Time Spent</span>
                  <div className="text-gray-900 dark:text-white font-normal">2.5 hours</div>
                </div>
              </div>
            </div>
          </div>

          {/* SaaS Development Track Progress */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center space-x-3 mb-4">
              <Icons.Cloud className="w-5 h-5 text-purple-600" />
              <h3 className="text-xs font-bold text-gray-900 dark:text-white">
                SaaS Development Track
              </h3>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-gray-600 dark:text-gray-300">Progress</span>
                <span className="text-gray-900 dark:text-white">8 of 632 lessons</span>
              </div>
              <ProgressBar value={8} max={632} showPercentage />
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-gray-600 dark:text-gray-300">Modules</span>
                  <div className="text-gray-900 dark:text-white font-normal">0 of 19</div>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-300">Time Spent</span>
                  <div className="text-gray-900 dark:text-white font-normal">4.2 hours</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Learning Overview
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600 mb-1">20</div>
              <div className="text-xs text-gray-600 dark:text-gray-300">Lessons Completed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600 mb-1">11</div>
              <div className="text-xs text-gray-600 dark:text-gray-300">Hours Learned</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600 mb-1">7</div>
              <div className="text-xs text-gray-600 dark:text-gray-300">Day Streak</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600 mb-1">2</div>
              <div className="text-xs text-gray-600 dark:text-gray-300">Certificates</div>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Recent Activity
          </h2>
          <div className="space-y-3">
            {[
              {
                title: "Prompt Engineering Principles",
                track: "AI Training",
                time: "2 hours ago",
                icon: Icons.Bot,
                color: "blue"
              },
              {
                title: "Database Fundamentals Overview",
                track: "Data Engineering",
                time: "1 day ago",
                icon: Icons.Database,
                color: "green"
              },
              {
                title: "AI Foundation & Tool Fluency",
                track: "AI Training",
                time: "2 days ago",
                icon: Icons.Bot,
                color: "blue"
              }
            ].map((activity, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg">
                <activity.icon className={`w-4 h-4 text-${activity.color}-600`} />
                <div className="flex-1">
                  <div className="text-xs font-normal text-gray-900 dark:text-white">
                    {activity.title}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">
                    {activity.track} • {activity.time}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </MainLayout>
  )
}