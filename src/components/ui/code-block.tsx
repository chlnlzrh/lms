'use client'

import dynamic from 'next/dynamic'

const SyntaxHighlighter = dynamic(
  () => import('react-syntax-highlighter').then(mod => mod.Prism),
  { 
    ssr: false,
    loading: () => null
  }
)

interface CodeBlockProps {
  language: string
  code: string
}

export function CodeBlock({ language, code }: CodeBlockProps) {
  return (
    <div className="my-4">
      <div className="bg-gray-800 text-gray-200 rounded-t-lg px-3 py-1 text-xs font-mono">
        {language}
      </div>
      <SyntaxHighlighter
        language={language}
        style={{
          'code[class*="language-"]': {
            color: '#d6deeb',
            fontFamily: 'Fira Code, Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace',
            fontSize: '12px',
            lineHeight: '1.4',
            textAlign: 'left',
            whiteSpace: 'pre',
            wordSpacing: 'normal',
            wordBreak: 'normal',
            wordWrap: 'normal',
            tabSize: '4',
            hyphens: 'none'
          },
          'pre[class*="language-"]': {
            color: '#d6deeb',
            fontFamily: 'Fira Code, Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace',
            fontSize: '12px',
            lineHeight: '1.4',
            textAlign: 'left',
            whiteSpace: 'pre',
            wordSpacing: 'normal',
            wordBreak: 'normal',
            wordWrap: 'normal',
            tabSize: '4',
            hyphens: 'none',
            background: '#1a1a1a',
            padding: '16px',
            margin: '0',
            overflow: 'auto'
          },
          comment: { color: '#8e9aaf' },
          prolog: { color: '#8e9aaf' },
          doctype: { color: '#8e9aaf' },
          cdata: { color: '#8e9aaf' },
          punctuation: { color: '#d6deeb' },
          property: { color: '#addb67' },
          tag: { color: '#addb67' },
          boolean: { color: '#addb67' },
          number: { color: '#addb67' },
          constant: { color: '#addb67' },
          symbol: { color: '#addb67' },
          deleted: { color: '#addb67' },
          selector: { color: '#ecc48d' },
          'attr-name': { color: '#ecc48d' },
          string: { color: '#ecc48d' },
          char: { color: '#ecc48d' },
          builtin: { color: '#ecc48d' },
          inserted: { color: '#ecc48d' },
          operator: { color: '#7fdbca' },
          entity: { color: '#7fdbca' },
          url: { color: '#7fdbca' },
          atrule: { color: '#c792ea' },
          'attr-value': { color: '#c792ea' },
          keyword: { color: '#c792ea' },
          function: { color: '#82aaff' },
          'class-name': { color: '#82aaff' },
          regex: { color: '#d6deeb' },
          important: { color: '#d6deeb', fontWeight: 'bold' },
          variable: { color: '#d6deeb' },
          bold: { fontWeight: 'bold' },
          italic: { fontStyle: 'italic' }
        }}
        customStyle={{
          background: '#1a1a1a',
          borderRadius: '0 0 8px 8px',
          fontSize: '12px',
          lineHeight: '1.4',
          margin: 0,
          padding: '16px'
        }}
        showLineNumbers={false}
        wrapLines={true}
        wrapLongLines={true}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}