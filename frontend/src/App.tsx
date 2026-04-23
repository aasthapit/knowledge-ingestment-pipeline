import { BrowserRouter, Routes, Route } from "react-router-dom"
import Layout from "@/components/Layout"
import Dashboard from "@/pages/Dashboard"
import Ingest from "@/pages/Ingest"
import Review from "@/pages/Review"
import Search from "@/pages/Search"
import Confluence from "@/pages/Confluence"
import KBHealth from "@/pages/KBHealth"
import Ledger from "@/pages/Ledger"
import Corpus from "@/pages/Corpus"
import Manifests from "@/pages/Manifests"
import Status from "@/pages/Status"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="ingest" element={<Ingest />} />
          <Route path="review" element={<Review />} />
          <Route path="search" element={<Search />} />
          <Route path="confluence" element={<Confluence />} />
          <Route path="health" element={<KBHealth />} />
          <Route path="ledger" element={<Ledger />} />
          <Route path="corpus" element={<Corpus />} />
          <Route path="manifests" element={<Manifests />} />
          <Route path="status" element={<Status />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
