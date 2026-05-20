// -*- C++ -*-
//
// Package:    Validation/MtdValidation
// Class:      EtlSimHitsValidation
//
/**\class EtlSimHitsValidation EtlSimHitsValidation.cc Validation/MtdValidation/plugins/EtlSimHitsValidation.cc

 Description: ETL SIM hits validation

 Implementation:
     [Notes on implementation]
*/

#include <string>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "SimFastTiming/FastTimingCommon/interface/MTDDigitizerTypes.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomUtil.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "MTDHit.h"

#include "DataFormats/Math/interface/angle_units.h"

struct EnteringTrackDiskSummary {
  struct HitInfo {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float tof = 0.f;
    float energyLoss = 0.f;
    int face = -1;
    int offsetTrackId = -999;
    unsigned int trackId = 0;
    unsigned int originalTrackId = 0;
    unsigned int detUnitId = 0;
    int zside = 0;
    int disc = 0;
  };

  std::array<int, 2> nSimHitsPerDisk{{0, 0}};
  std::array<std::array<int, 2>, 2> nSimHitsPerDiskFace{{{{0, 0}}, {{0, 0}}}};
  std::array<std::array<int, 2>, 2> nSimHitsPerDiskFaceNoOffset4{{{{0, 0}}, {{0, 0}}}};
  std::array<int, 2> nFrontHitsPerDisk{{0, 0}};
  std::array<int, 2> nBackscatterHitsPerDisk{{0, 0}};
  std::array<std::vector<HitInfo>, 2> hitsPerDisk;

  float trackPtAtProduction = -1.f;
  bool hasTrackPtAtProduction = false;

  void setTrackPtAtProduction(float pt) {
    trackPtAtProduction = pt;
    hasTrackPtAtProduction = true;
  }

  void addHit(int disc,
              int face,
              float x,
              float y,
              float z,
              float tof,
              float energyLoss,
              int offsetTrackId,
              unsigned int trackId,
              unsigned int originalTrackId,
              unsigned int detUnitId,
              int zside) {
    if (disc < 1 || disc > 2)
      return;

    const int diskIndex = disc - 1;
    ++nSimHitsPerDisk[diskIndex];

    if (face == 0 || face == 1) {
      ++nSimHitsPerDiskFace[diskIndex][face];

      if (offsetTrackId != 4) {
        ++nSimHitsPerDiskFaceNoOffset4[diskIndex][face];
      }
    }

    if (offsetTrackId == 0) {
      ++nFrontHitsPerDisk[diskIndex];
    } else if (offsetTrackId == 4) {
      ++nBackscatterHitsPerDisk[diskIndex];
    }

    hitsPerDisk[diskIndex].push_back({x,
                                      y,
                                      z,
                                      tof,
                                      energyLoss,
                                      face,
                                      offsetTrackId,
                                      trackId,
                                      originalTrackId,
                                      detUnitId,
                                      zside,
                                      disc});
  }
};

struct DispersionResult {
  float maxPairwiseXY = 0.f;
  float timeSpread = 0.f;
  int latestOffsetTrackId = -999;
  int maxPairIndex1 = -1;
  int maxPairIndex2 = -1;
};

static DispersionResult computeDispersion(const std::vector<EnteringTrackDiskSummary::HitInfo>& hits) {
  DispersionResult result;

  const size_t nHits = hits.size();
  if (nHits == 0) {
    return result;
  }

  float minTof = std::numeric_limits<float>::max();
  float maxTof = -std::numeric_limits<float>::max();

  for (const auto& hit : hits) {
    if (hit.tof < minTof)
      minTof = hit.tof;
    if (hit.tof > maxTof) {
      maxTof = hit.tof;
      result.latestOffsetTrackId = hit.offsetTrackId;
    }
  }

  result.timeSpread = maxTof - minTof;

  for (size_t i = 0; i < nHits; ++i) {
    for (size_t j = i + 1; j < nHits; ++j) {
      const float dx = hits[i].x - hits[j].x;
      const float dy = hits[i].y - hits[j].y;
      const float distXY = std::sqrt(dx * dx + dy * dy);

      if (distXY > result.maxPairwiseXY) {
        result.maxPairwiseXY = distXY;
        result.maxPairIndex1 = static_cast<int>(i);
        result.maxPairIndex2 = static_cast<int>(j);
      }
    }
  }

  return result;
}


static void printLargeSpaceDispersionHits(const edm::Event& iEvent,
                                           int originalTrackId,
                                           int diskIndex,
                                           const std::vector<EnteringTrackDiskSummary::HitInfo>& hits,
                                           const DispersionResult& dispersion) {
  if (dispersion.maxPairwiseXY <= 90.f)
    return;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "[Large space dispersion] "
            << "run=" << iEvent.id().run()
            << ", lumi=" << iEvent.id().luminosityBlock()
            << ", event=" << iEvent.id().event()
            << ", originalTrackId=" << originalTrackId
            << ", disk=D" << (diskIndex + 1)
            << ", nHits=" << hits.size()
            << ", maxPairwiseXY=" << dispersion.maxPairwiseXY << " cm"
            << ", maxPairIndices=(" << dispersion.maxPairIndex1
            << ", " << dispersion.maxPairIndex2 << ")"
            << std::endl;

  for (size_t ihit = 0; ihit < hits.size(); ++ihit) {
    const auto& hit = hits[ihit];
    std::cout << "  hit[" << ihit << "]"
              << " trackId=" << hit.trackId
              << " originalTrackId=" << hit.originalTrackId
              << " offsetTrackId=" << hit.offsetTrackId
              << " detUnitId=" << hit.detUnitId
              << " zside=" << hit.zside
              << " disc=" << hit.disc
              << " face=" << hit.face
              << " x=" << hit.x << " cm"
              << " y=" << hit.y << " cm"
              << " z=" << hit.z << " cm"
              << " tof=" << hit.tof << " ns"
              << " energyLoss=" << hit.energyLoss
              << std::endl;
  }
}

static DispersionResult computeDispersionOffset0Only(const std::vector<EnteringTrackDiskSummary::HitInfo>& hits) {
  std::vector<EnteringTrackDiskSummary::HitInfo> selectedHits;
  selectedHits.reserve(hits.size());

  for (const auto& hit : hits) {
    if (hit.offsetTrackId == 0) {
      selectedHits.push_back(hit);
    }
  }

  return computeDispersion(selectedHits);
}

struct PerDiskSummaryValues {
  int nHits = 0;
  int nFrontHits = 0;
  int nBackscatterHits = 0;
  int nFace0Hits = 0;
  int nFace1Hits = 0;
  int nFace0HitsNoOffset4 = 0;
  int nFace1HitsNoOffset4 = 0;
  float pt = -1.f;
  bool hasPt = false;
  float spaceXY = 0.f;
  float timeSpread = 0.f;
  float timeSpreadOffset0Only = 0.f;
};

static void fillDiskSummaryHistograms(const PerDiskSummaryValues& v,
                                      dqm::impl::MonitorElement* hNHitVsPt,
                                      dqm::impl::MonitorElement* hFace2VsFace1,
                                      dqm::impl::MonitorElement* hFace2VsFace1NoOffset4,
                                      dqm::impl::MonitorElement* hBackScatterVsFront,
                                      dqm::impl::MonitorElement* hSpaceXY,
                                      dqm::impl::MonitorElement* hTime,
                                      dqm::impl::MonitorElement* hTimeOffset0Only,
                                      dqm::impl::MonitorElement* hNHitVsSpaceXY,
                                      dqm::impl::MonitorElement* hNHitVsTime) {
  if (v.nHits <= 0)
    return;

  if (v.hasPt) {
    hNHitVsPt->Fill(v.nHits, v.pt);
  }

  hFace2VsFace1->Fill(v.nFace0Hits, v.nFace1Hits);
  hFace2VsFace1NoOffset4->Fill(v.nFace0HitsNoOffset4, v.nFace1HitsNoOffset4);
  hBackScatterVsFront->Fill(v.nFrontHits, v.nBackscatterHits);
  hSpaceXY->Fill(v.spaceXY);
  hTime->Fill(v.timeSpread);
  hNHitVsSpaceXY->Fill(v.nHits, v.spaceXY);
  hNHitVsTime->Fill(v.nHits, v.timeSpread);

  if (v.nFrontHits > 0) {
    hTimeOffset0Only->Fill(v.timeSpreadOffset0Only);
  }
}

static PerDiskSummaryValues makePerDiskSummaryValues(const EnteringTrackDiskSummary& summary,
                                                     int diskIndex,
                                                     const DispersionResult& dispersion,
                                                     const DispersionResult& dispersionOffset0Only) {
  PerDiskSummaryValues values;

  values.nHits = summary.nSimHitsPerDisk[diskIndex];
  values.nFrontHits = summary.nFrontHitsPerDisk[diskIndex];
  values.nBackscatterHits = summary.nBackscatterHitsPerDisk[diskIndex];
  values.nFace0Hits = summary.nSimHitsPerDiskFace[diskIndex][0];
  values.nFace1Hits = summary.nSimHitsPerDiskFace[diskIndex][1];
  values.nFace0HitsNoOffset4 = summary.nSimHitsPerDiskFaceNoOffset4[diskIndex][0];
  values.nFace1HitsNoOffset4 = summary.nSimHitsPerDiskFaceNoOffset4[diskIndex][1];
  values.pt = summary.trackPtAtProduction;
  values.hasPt = summary.hasTrackPtAtProduction;
  values.spaceXY = dispersion.maxPairwiseXY;
  values.timeSpread = dispersion.timeSpread;
  values.timeSpreadOffset0Only = dispersionOffset0Only.timeSpread;

  return values;
}

class EtlSimHitsValidation : public DQMEDAnalyzer {
public:
  explicit EtlSimHitsValidation(const edm::ParameterSet&);
  ~EtlSimHitsValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const std::string folder_;
  const float hitMinEnergy2Dis_;
  const bool optionalPlots_;

  edm::EDGetTokenT<CrossingFrame<PSimHit>> etlSimHitsToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;

  MonitorElement* meNhits_[4];
  MonitorElement* meNtrkPerCell_[4];

  MonitorElement* meHitEnergy_[4];
  MonitorElement* meHitTime_[4];

  MonitorElement* meHitXlocal_[4];
  MonitorElement* meHitYlocal_[4];
  MonitorElement* meHitZlocal_[4];

  MonitorElement* meOccupancy_[4];

  MonitorElement* meHitX_[4];
  MonitorElement* meHitY_[4];
  MonitorElement* meHitZ_[4];
  MonitorElement* meHitPhi_[4];
  MonitorElement* meHitEta_[4];

  MonitorElement* meHitTvsE_[4];
  MonitorElement* meHitEvsPhi_[4];
  MonitorElement* meHitEvsEta_[4];
  MonitorElement* meHitTvsPhi_[4];
  MonitorElement* meHitTvsEta_[4];

  MonitorElement* meNSimHitsPerEnteringTrackD1_ = nullptr;
  MonitorElement* meNSimHitsPerEnteringTrackD2_ = nullptr;
  MonitorElement* meNSimHitsFace2VsFace1D1_ = nullptr;
  MonitorElement* meNSimHitsFace2VsFace1D2_ = nullptr;
  MonitorElement* meNSimHitsFace2VsFace1D1NoOffset4_ = nullptr;
  MonitorElement* meNSimHitsFace2VsFace1D2NoOffset4_ = nullptr;
  MonitorElement* meNBackScatterVsFrontHitsD1_ = nullptr;
  MonitorElement* meNBackScatterVsFrontHitsD2_ = nullptr;

  MonitorElement* meSpaceDispersionXYD1_ = nullptr;
  MonitorElement* meSpaceDispersionXYD2_ = nullptr;
  MonitorElement* meTimeDispersionD1_ = nullptr;
  MonitorElement* meTimeDispersionD2_ = nullptr;
  MonitorElement* meTimeDispersionLatestOffset4D1_ = nullptr;
  MonitorElement* meTimeDispersionLatestOffset4D2_ = nullptr;
  MonitorElement* meNHitVsSpaceDispersionXYD1_ = nullptr;
  MonitorElement* meNHitVsSpaceDispersionXYD2_ = nullptr;
  MonitorElement* meNHitVsTimeDispersionD1_ = nullptr;
  MonitorElement* meNHitVsTimeDispersionD2_ = nullptr;

  MonitorElement* meHitThetaEntryD1_[3];
  MonitorElement* meHitThetaEntryD2_[3];

  static constexpr int n_bin_Eta = 3;
  static constexpr double eta_bins_edges[n_bin_Eta + 1] = {1.5, 2.1, 2.5, 3.0};
};

EtlSimHitsValidation::EtlSimHitsValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      hitMinEnergy2Dis_(iConfig.getParameter<double>("hitMinimumEnergy2Dis")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")) {
  etlSimHitsToken_ = consumes<CrossingFrame<PSimHit>>(iConfig.getParameter<edm::InputTag>("inputTag"));
  simTracksToken_ = consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simTrackTag"));
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
}

EtlSimHitsValidation::~EtlSimHitsValidation() {}

void EtlSimHitsValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;

  using namespace mtd;
  using namespace std;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();

  MTDGeomUtil geomUtil;
  geomUtil.setGeometry(geom);

  auto etlSimHitsHandle = makeValid(iEvent.getHandle(etlSimHitsToken_));
  MixCollection<PSimHit> etlSimHits(etlSimHitsHandle.product());

  auto simTracksHandle = makeValid(iEvent.getHandle(simTracksToken_));
  const edm::SimTrackContainer& simTracks = *simTracksHandle;

  std::unordered_map<unsigned int, float> simTrackPtAtProduction;
  for (auto const& simTrack : simTracks) {
    simTrackPtAtProduction[simTrack.trackId()] = simTrack.momentum().pt();
  }

  std::unordered_map<mtd_digitizer::MTDCellId, MTDHit> m_etlHits[4];
  std::unordered_map<mtd_digitizer::MTDCellId, std::set<int>> m_etlTrkPerCell[4];

  std::map<int, EnteringTrackDiskSummary> enteringTracks;

  int idet = 999;
  size_t index(0);

  for (auto const& simHit : etlSimHits) {
    index++;
    LogDebug("EtlSimHitsValidation") << "SimHit # " << index << " detId " << simHit.detUnitId() << " ene "
                                     << simHit.energyLoss() << " tof " << simHit.tof() << " tId " << simHit.trackId();

    ETLDetId id = simHit.detUnitId();
    if ((id.zside() == -1) && (id.nDisc() == 1)) {
      idet = 0;
    } else if ((id.zside() == -1) && (id.nDisc() == 2)) {
      idet = 1;
    } else if ((id.zside() == 1) && (id.nDisc() == 1)) {
      idet = 2;
    } else if ((id.zside() == 1) && (id.nDisc() == 2)) {
      idet = 3;
    } else {
      edm::LogWarning("EtlSimHitsValidation") << "Unknown ETL DetId configuration: " << id;
      continue;
    }

    const auto& position = simHit.localPosition();

    DetId geoIdForThisHit = id.geographicalId();
    const MTDGeomDet* thedetForThisHit = geom->idToDet(geoIdForThisHit);
    if (thedetForThisHit == nullptr)
      throw cms::Exception("EtlSimHitsValidation") << "GeographicalID: " << std::hex << geoIdForThisHit.rawId() << " ("
                                                   << id.rawId() << ") is invalid!" << std::dec << std::endl;

    Local3DPoint localPointForThisHit(convertMmToCm(position.x()),
                                      convertMmToCm(position.y()),
                                      convertMmToCm(position.z()));
    const auto& globalPointForThisHit = thedetForThisHit->toGlobal(localPointForThisHit);

    if (simHit.offsetTrackId() != 0 && simHit.offsetTrackId() != 4) {
      throw cms::Exception("EtlSimHitsValidation")
          << "Unexpected offsetTrackId " << simHit.offsetTrackId()
          << " for originalTrackId " << simHit.originalTrackId()
          << ", detId " << simHit.detUnitId();
    }

    auto& enteringTrackSummary = enteringTracks[simHit.originalTrackId()];
    enteringTrackSummary.addHit(id.nDisc(),
                                id.discSide(),
                                globalPointForThisHit.x(),
                                globalPointForThisHit.y(),
                                globalPointForThisHit.z(),
                                simHit.tof(),
                                simHit.energyLoss(),
                                simHit.offsetTrackId(),
                                simHit.trackId(),
                                simHit.originalTrackId(),
                                simHit.detUnitId(),
                                id.zside());

    if (!enteringTrackSummary.hasTrackPtAtProduction) {
      auto itPt = simTrackPtAtProduction.find(static_cast<unsigned int>(simHit.originalTrackId()));
      if (itPt != simTrackPtAtProduction.end()) {
        enteringTrackSummary.setTrackPtAtProduction(itPt->second);
      }
    }

    LocalPoint simscaled(convertMmToCm(position.x()), convertMmToCm(position.y()), convertMmToCm(position.z()));
    std::pair<uint8_t, uint8_t> pixel = geomUtil.pixelInModule(id, simscaled);

    mtd_digitizer::MTDCellId pixelId(id.rawId(), pixel.first, pixel.second);
    m_etlTrkPerCell[idet][pixelId].insert(simHit.trackId());
    auto simHitIt = m_etlHits[idet].emplace(pixelId, MTDHit()).first;

    (simHitIt->second).energy += convertUnitsTo(0.001_MeV, simHit.energyLoss());

    if ((simHitIt->second).time == 0 || simHit.tof() < (simHitIt->second).time) {
      (simHitIt->second).time = simHit.tof();

      auto hit_pos = simHit.localPosition();
      (simHitIt->second).x = hit_pos.x();
      (simHitIt->second).y = hit_pos.y();
      (simHitIt->second).z = hit_pos.z();

      if (simHit.offsetTrackId() == 0) {
        if (simHit.exitPoint() != simHit.entryPoint()) {
          (simHitIt->second).thetaAtEntry =
              angle_units::operators::convertRadToDeg((simHit.exitPoint() - simHit.entryPoint()).bareTheta());
          if (id.discSide() == 1) {
            (simHitIt->second).thetaAtEntry = 180. - (simHitIt->second).thetaAtEntry;
          }
        }
      } else {
        (simHitIt->second).thetaAtEntry = -90.;
      }
    }
    LogDebug("EtlSimHitsValidation") << "Registered in idet " << idet;

  }  // simHit loop

  for (int idet = 0; idet < 4; ++idet) {
    meNhits_[idet]->Fill(m_etlHits[idet].size());
    LogDebug("EtlSimHitsValidation") << "idet " << idet << " #hits " << m_etlHits[idet].size();

    for (auto const& hit : m_etlTrkPerCell[idet]) {
      meNtrkPerCell_[idet]->Fill((hit.second).size());
    }

    for (auto const& hit : m_etlHits[idet]) {
      double weight = 1.0;
      if ((hit.second).energy < hitMinEnergy2Dis_)
        continue;

      ETLDetId detId;
      detId = hit.first.detid_;
      DetId geoId = detId.geographicalId();
      const MTDGeomDet* thedet = geom->idToDet(geoId);
      if (thedet == nullptr)
        throw cms::Exception("EtlSimHitsValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                     << detId.rawId() << ") is invalid!" << std::dec << std::endl;

      Local3DPoint local_point(
          convertMmToCm((hit.second).x), convertMmToCm((hit.second).y), convertMmToCm((hit.second).z));
      const auto& global_point = thedet->toGlobal(local_point);

      if (detId.discSide() == 1) {
        weight = -weight;
      }

      meHitEnergy_[idet]->Fill((hit.second).energy);
      meHitTime_[idet]->Fill((hit.second).time);
      meHitXlocal_[idet]->Fill((hit.second).x);
      meHitYlocal_[idet]->Fill((hit.second).y);
      meHitZlocal_[idet]->Fill((hit.second).z);
      meOccupancy_[idet]->Fill(global_point.x(), global_point.y(), weight);
      meHitX_[idet]->Fill(global_point.x());
      meHitY_[idet]->Fill(global_point.y());
      meHitZ_[idet]->Fill(global_point.z());
      meHitPhi_[idet]->Fill(global_point.phi());
      meHitEta_[idet]->Fill(global_point.eta());
      meHitTvsE_[idet]->Fill((hit.second).energy, (hit.second).time);
      meHitEvsPhi_[idet]->Fill(global_point.phi(), (hit.second).energy);
      meHitEvsEta_[idet]->Fill(global_point.eta(), (hit.second).energy);
      meHitTvsPhi_[idet]->Fill(global_point.phi(), (hit.second).time);
      meHitTvsEta_[idet]->Fill(global_point.eta(), (hit.second).time);

      if (optionalPlots_) {
        if ((hit.second).thetaAtEntry > 0.) {
          std::size_t ibin(0);
          for (size_t i = 0; i < n_bin_Eta; i++) {
            if (std::abs(global_point.eta()) >= eta_bins_edges[i] &&
                std::abs(global_point.eta()) < eta_bins_edges[i + 1]) {
              ibin = i;
              break;
            }
          }
          if (idet == 0 || idet == 2) {
            meHitThetaEntryD1_[ibin]->Fill((hit.second).thetaAtEntry);
          } else {
            meHitThetaEntryD2_[ibin]->Fill((hit.second).thetaAtEntry);
          }
        }
      }
    }
  }

  for (const auto& enteringTrack : enteringTracks) {
    const auto& summary = enteringTrack.second;

    if (summary.nSimHitsPerDisk[0] > 0) {
      const auto dispersionD1 = computeDispersion(summary.hitsPerDisk[0]);
      printLargeSpaceDispersionHits(iEvent, enteringTrack.first, 0, summary.hitsPerDisk[0], dispersionD1);
      const auto dispersionD1Offset0Only = computeDispersionOffset0Only(summary.hitsPerDisk[0]);
      const auto valuesD1 = makePerDiskSummaryValues(summary, 0, dispersionD1, dispersionD1Offset0Only);

      fillDiskSummaryHistograms(valuesD1,
                                meNSimHitsPerEnteringTrackD1_,
                                meNSimHitsFace2VsFace1D1_,
                                meNSimHitsFace2VsFace1D1NoOffset4_,
                                meNBackScatterVsFrontHitsD1_,
                                meSpaceDispersionXYD1_,
                                meTimeDispersionD1_,
                                meTimeDispersionLatestOffset4D1_,
                                meNHitVsSpaceDispersionXYD1_,
                                meNHitVsTimeDispersionD1_);
    }

    if (summary.nSimHitsPerDisk[1] > 0) {
      const auto dispersionD2 = computeDispersion(summary.hitsPerDisk[1]);
      printLargeSpaceDispersionHits(iEvent, enteringTrack.first, 1, summary.hitsPerDisk[1], dispersionD2);
      const auto dispersionD2Offset0Only = computeDispersionOffset0Only(summary.hitsPerDisk[1]);
      const auto valuesD2 = makePerDiskSummaryValues(summary, 1, dispersionD2, dispersionD2Offset0Only);

      fillDiskSummaryHistograms(valuesD2,
                                meNSimHitsPerEnteringTrackD2_,
                                meNSimHitsFace2VsFace1D2_,
                                meNSimHitsFace2VsFace1D2NoOffset4_,
                                meNBackScatterVsFrontHitsD2_,
                                meSpaceDispersionXYD2_,
                                meTimeDispersionD2_,
                                meTimeDispersionLatestOffset4D2_,
                                meNHitVsSpaceDispersionXYD2_,
                                meNHitVsTimeDispersionD2_);
    }
  }
}

void EtlSimHitsValidation::bookHistograms(DQMStore::IBooker& ibook,
                                          edm::Run const& run,
                                          edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  meNhits_[0] = ibook.book1D("EtlNhitsZnegD1",
                             "Number of ETL cells with SIM hits (-Z, Single(topo1D)/First(topo2D) disk);N_{ETL cells}",
                             100,
                             0.,
                             5000.);
  meNhits_[1] = ibook.book1D(
      "EtlNhitsZnegD2", "Number of ETL cells with SIM hits (-Z, Second disk);N_{ETL cells}", 100, 0., 5000.);
  meNhits_[2] = ibook.book1D("EtlNhitsZposD1",
                             "Number of ETL cells with SIM hits (+Z, Single(topo1D)/First(topo2D) disk);N_{ETL cells}",
                             100,
                             0.,
                             5000.);
  meNhits_[3] = ibook.book1D(
      "EtlNhitsZposD2", "Number of ETL cells with SIM hits (+Z, Second Disk);N_{ETL cells}", 100, 0., 5000.);
  meNtrkPerCell_[0] = ibook.book1D("EtlNtrkPerCellZnegD1",
                                   "Number of tracks per ETL sensor (-Z, Single(topo1D)/First(topo2D) disk);N_{trk}",
                                   10,
                                   0.,
                                   10.);
  meNtrkPerCell_[1] =
      ibook.book1D("EtlNtrkPerCellZnegD2", "Number of tracks per ETL sensor (-Z, Second disk);N_{trk}", 10, 0., 10.);
  meNtrkPerCell_[2] = ibook.book1D("EtlNtrkPerCellZposD1",
                                   "Number of tracks per ETL sensor (+Z, Single(topo1D)/First(topo2D) disk);N_{trk}",
                                   10,
                                   0.,
                                   10.);
  meNtrkPerCell_[3] =
      ibook.book1D("EtlNtrkPerCellZposD2", "Number of tracks per ETL sensor (+Z, Second disk);N_{trk}", 10, 0., 10.);

  meHitEnergy_[0] = ibook.book1D(
      "EtlHitEnergyZnegD1", "ETL SIM hits energy (-Z, Single(topo1D)/First(topo2D) disk);E_{SIM} [MeV]", 100, 0., 1.5);
  meHitEnergy_[1] =
      ibook.book1D("EtlHitEnergyZnegD2", "ETL SIM hits energy (-Z, Second disk);E_{SIM} [MeV]", 100, 0., 1.5);
  meHitEnergy_[2] = ibook.book1D(
      "EtlHitEnergyZposD1", "ETL SIM hits energy (+Z, Single(topo1D)/First(topo2D) disk);E_{SIM} [MeV]", 100, 0., 1.5);
  meHitEnergy_[3] =
      ibook.book1D("EtlHitEnergyZposD2", "ETL SIM hits energy (+Z, Second disk);E_{SIM} [MeV]", 100, 0., 1.5);

  meHitTime_[0] = ibook.book1D(
      "EtlHitTimeZnegD1", "ETL SIM hits ToA (-Z, Single(topo1D)/First(topo2D) disk);ToA_{SIM} [ns]", 100, 0., 25.);
  meHitTime_[1] = ibook.book1D("EtlHitTimeZnegD2", "ETL SIM hits ToA (-Z, Second disk);ToA_{SIM} [ns]", 100, 0., 25.);
  meHitTime_[2] = ibook.book1D(
      "EtlHitTimeZposD1", "ETL SIM hits ToA (+Z, Single(topo1D)/First(topo2D) disk);ToA_{SIM} [ns]", 100, 0., 25.);
  meHitTime_[3] = ibook.book1D("EtlHitTimeZposD2", "ETL SIM hits ToA (+Z, Second disk);ToA_{SIM} [ns]", 100, 0., 25.);

  meHitXlocal_[0] = ibook.book1D("EtlHitXlocalZnegD1",
                                 "ETL SIM local X (-Z, Single(topo1D)/First(topo2D) disk);X_{SIM}^{LOC} [mm]",
                                 100,
                                 -25.,
                                 25.);
  meHitXlocal_[1] =
      ibook.book1D("EtlHitXlocalZnegD2", "ETL SIM local X (-Z, Second disk);X_{SIM}^{LOC} [mm]", 100, -25., 25.);
  meHitXlocal_[2] = ibook.book1D("EtlHitXlocalZposD1",
                                 "ETL SIM local X (+Z, Single(topo1D)/First(topo2D) disk);X_{SIM}^{LOC} [mm]",
                                 100,
                                 -25.,
                                 25.);
  meHitXlocal_[3] =
      ibook.book1D("EtlHitXlocalZposD2", "ETL SIM local X (+Z, Second disk);X_{SIM}^{LOC} [mm]", 100, -25., 25.);

  meHitYlocal_[0] = ibook.book1D("EtlHitYlocalZnegD1",
                                 "ETL SIM local Y (-Z, Single(topo1D)/First(topo2D) disk);Y_{SIM}^{LOC} [mm]",
                                 100,
                                 -48.,
                                 48.);
  meHitYlocal_[1] =
      ibook.book1D("EtlHitYlocalZnegD2", "ETL SIM local Y (-Z, Second Disk);Y_{SIM}^{LOC} [mm]", 100, -48., 48.);
  meHitYlocal_[2] = ibook.book1D("EtlHitYlocalZposD1",
                                 "ETL SIM local Y (+Z, Single(topo1D)/First(topo2D) disk);Y_{SIM}^{LOC} [mm]",
                                 100,
                                 -48.,
                                 48.);
  meHitYlocal_[3] =
      ibook.book1D("EtlHitYlocalZposD2", "ETL SIM local Y (+Z, Second disk);Y_{SIM}^{LOC} [mm]", 100, -48., 48.);
  meHitZlocal_[0] = ibook.book1D("EtlHitZlocalZnegD1",
                                 "ETL SIM local Z (-Z, Single(topo1D)/First(topo2D) disk);Z_{SIM}^{LOC} [mm]",
                                 80,
                                 -0.16,
                                 0.16);
  meHitZlocal_[1] =
      ibook.book1D("EtlHitZlocalZnegD2", "ETL SIM local Z (-Z, Second disk);Z_{SIM}^{LOC} [mm]", 80, -0.16, 0.16);
  meHitZlocal_[2] = ibook.book1D("EtlHitZlocalZposD1",
                                 "ETL SIM local Z (+Z, Single(topo1D)/First(topo2D) disk);Z_{SIM}^{LOC} [mm]",
                                 80,
                                 -0.16,
                                 0.16);
  meHitZlocal_[3] =
      ibook.book1D("EtlHitZlocalZposD2", "ETL SIM local Z (+Z, Second disk);Z_{SIM}^{LOC} [mm]", 80, -0.16, 0.16);

  meOccupancy_[0] =
      ibook.book2D("EtlOccupancyZnegD1",
                   "ETL SIM hits occupancy (-Z, Single(topo1D)/First(topo2D) disk);X_{SIM} [cm];Y_{SIM} [cm]",
                   135,
                   -135.,
                   135.,
                   135,
                   -135.,
                   135.);
  meOccupancy_[1] = ibook.book2D("EtlOccupancyZnegD2",
                                 "ETL SIM hits occupancy (-Z, Second disk);X_{SIM} [cm];Y_{SIM} [cm]",
                                 135,
                                 -135.,
                                 135.,
                                 135,
                                 -135.,
                                 135.);
  meOccupancy_[2] =
      ibook.book2D("EtlOccupancyZposD1",
                   "ETL SIM hits occupancy (+Z, Single(topo1D)/First(topo2D) disk);X_{SIM} [cm];Y_{SIM} [cm]",
                   135,
                   -135.,
                   135.,
                   135,
                   -135.,
                   135.);
  meOccupancy_[3] = ibook.book2D("EtlOccupancyZposD2",
                                 "ETL SIM hits occupancy (+Z, Second disk);X_{SIM} [cm];Y_{SIM} [cm]",
                                 135,
                                 -135.,
                                 135.,
                                 135,
                                 -135.,
                                 135.);

  meHitX_[0] = ibook.book1D(
      "EtlHitXZnegD1", "ETL SIM hits X (+Z, Single(topo1D)/First(topo2D) disk);X_{SIM} [cm]", 100, -130., 130.);
  meHitX_[1] = ibook.book1D("EtlHitXZnegD2", "ETL SIM hits X (-Z, Second disk);X_{SIM} [cm]", 100, -130., 130.);
  meHitX_[2] = ibook.book1D(
      "EtlHitXZposD1", "ETL SIM hits X (+Z, Single(topo1D)/First(topo2D) disk);X_{SIM} [cm]", 100, -130., 130.);
  meHitX_[3] = ibook.book1D("EtlHitXZposD2", "ETL SIM hits X (+Z, Second disk);X_{SIM} [cm]", 100, -130., 130.);
  meHitY_[0] = ibook.book1D(
      "EtlHitYZnegD1", "ETL SIM hits Y (-Z, Single(topo1D)/First(topo2D) disk);Y_{SIM} [cm]", 100, -130., 130.);
  meHitY_[1] = ibook.book1D("EtlHitYZnegD2", "ETL SIM hits Y (-Z, Second disk);Y_{SIM} [cm]", 100, -130., 130.);
  meHitY_[2] = ibook.book1D(
      "EtlHitYZposD1", "ETL SIM hits Y (+Z, Single(topo1D)/First(topo2D) disk);Y_{SIM} [cm]", 100, -130., 130.);
  meHitY_[3] = ibook.book1D("EtlHitYZposD2", "ETL SIM hits Y (+Z, Second disk);Y_{SIM} [cm]", 100, -130., 130.);
  meHitZ_[0] = ibook.book1D(
      "EtlHitZZnegD1", "ETL SIM hits Z (-Z, Single(topo1D)/First(topo2D) disk);Z_{SIM} [cm]", 100, -302., -298.);
  meHitZ_[1] = ibook.book1D("EtlHitZZnegD2", "ETL SIM hits Z (-Z, Second disk);Z_{SIM} [cm]", 100, -304., -300.);
  meHitZ_[2] = ibook.book1D(
      "EtlHitZZposD1", "ETL SIM hits Z (+Z, Single(topo1D)/First(topo2D) disk);Z_{SIM} [cm]", 100, 298., 302.);
  meHitZ_[3] = ibook.book1D("EtlHitZZposD2", "ETL SIM hits Z (+Z, Second disk);Z_{SIM} [cm]", 100, 300., 304.);

  meHitPhi_[0] = ibook.book1D(
      "EtlHitPhiZnegD1", "ETL SIM hits #phi (-Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad]", 100, -3.15, 3.15);
  meHitPhi_[1] =
      ibook.book1D("EtlHitPhiZnegD2", "ETL SIM hits #phi (-Z, Second disk);#phi_{SIM} [rad]", 100, -3.15, 3.15);
  meHitPhi_[2] = ibook.book1D(
      "EtlHitPhiZposD1", "ETL SIM hits #phi (+Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad]", 100, -3.15, 3.15);
  meHitPhi_[3] =
      ibook.book1D("EtlHitPhiZposD2", "ETL SIM hits #phi (+Z, Second disk);#phi_{SIM} [rad]", 100, -3.15, 3.15);
  meHitEta_[0] = ibook.book1D(
      "EtlHitEtaZnegD1", "ETL SIM hits #eta (-Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM}", 100, -3.2, -1.56);
  meHitEta_[1] = ibook.book1D("EtlHitEtaZnegD2", "ETL SIM hits #eta (-Z, Second disk);#eta_{SIM}", 100, -3.2, -1.56);
  meHitEta_[2] = ibook.book1D(
      "EtlHitEtaZposD1", "ETL SIM hits #eta (+Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM}", 100, 1.56, 3.2);
  meHitEta_[3] = ibook.book1D("EtlHitEtaZposD2", "ETL SIM hits #eta (+Z, Second disk);#eta_{SIM}", 100, 1.56, 3.2);

  meHitTvsE_[0] =
      ibook.bookProfile("EtlHitTvsEZnegD1",
                        "ETL SIM time vs energy (-Z, Single(topo1D)/First(topo2D) disk);E_{SIM} [MeV];T_{SIM} [ns]",
                        50,
                        0.,
                        2.,
                        0.,
                        100.);
  meHitTvsE_[1] = ibook.bookProfile(
      "EtlHitTvsEZnegD2", "ETL SIM time vs energy (-Z, Second disk);E_{SIM} [MeV];T_{SIM} [ns]", 50, 0., 2., 0., 100.);
  meHitTvsE_[2] =
      ibook.bookProfile("EtlHitTvsEZposD1",
                        "ETL SIM time vs energy (+Z, Single(topo1D)/First(topo2D) disk);E_{SIM} [MeV];T_{SIM} [ns]",
                        50,
                        0.,
                        2.,
                        0.,
                        100.);
  meHitTvsE_[3] = ibook.bookProfile(
      "EtlHitTvsEZposD2", "ETL SIM time vs energy (+Z, Second disk);E_{SIM} [MeV];T_{SIM} [ns]", 50, 0., 2., 0., 100.);

  meHitEvsPhi_[0] =
      ibook.bookProfile("EtlHitEvsPhiZnegD1",
                        "ETL SIM energy vs #phi (-Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad];E_{SIM} [MeV]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        100.);
  meHitEvsPhi_[1] = ibook.bookProfile("EtlHitEvsPhiZnegD2",
                                      "ETL SIM energy vs #phi (-Z, Second disk);#phi_{SIM} [rad];E_{SIM} [MeV]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      100.);
  meHitEvsPhi_[2] =
      ibook.bookProfile("EtlHitEvsPhiZposD1",
                        "ETL SIM energy vs #phi (+Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad];E_{SIM} [MeV]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        100.);
  meHitEvsPhi_[3] = ibook.bookProfile("EtlHitEvsPhiZposD2",
                                      "ETL SIM energy vs #phi (+Z, Second disk);#phi_{SIM} [rad];E_{SIM} [MeV]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      100.);

  meHitEvsEta_[0] =
      ibook.bookProfile("EtlHitEvsEtaZnegD1",
                        "ETL SIM energy vs #eta (-Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM};E_{SIM} [MeV]",
                        50,
                        -3.2,
                        -1.56,
                        0.,
                        100.);
  meHitEvsEta_[1] = ibook.bookProfile("EtlHitEvsEtaZnegD2",
                                      "ETL SIM energy vs #eta (-Z, Second disk);#eta_{SIM};E_{SIM} [MeV]",
                                      50,
                                      -3.2,
                                      -1.56,
                                      0.,
                                      100.);
  meHitEvsEta_[2] =
      ibook.bookProfile("EtlHitEvsEtaZposD1",
                        "ETL SIM energy vs #eta (+Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM};E_{SIM} [MeV]",
                        50,
                        1.56,
                        3.2,
                        0.,
                        100.);
  meHitEvsEta_[3] = ibook.bookProfile("EtlHitEvsEtaZposD2",
                                      "ETL SIM energy vs #eta (+Z, Second disk);#eta_{SIM};E_{SIM} [MeV]",
                                      50,
                                      1.56,
                                      3.2,
                                      0.,
                                      100.);

  meHitTvsPhi_[0] =
      ibook.bookProfile("EtlHitTvsPhiZnegD1",
                        "ETL SIM time vs #phi (-Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad];T_{SIM} [ns]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        100.);
  meHitTvsPhi_[1] = ibook.bookProfile("EtlHitTvsPhiZnegD2",
                                      "ETL SIM time vs #phi (-Z, Second disk);#phi_{SIM} [rad];T_{SIM} [ns]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      100.);
  meHitTvsPhi_[2] =
      ibook.bookProfile("EtlHitTvsPhiZposD1",
                        "ETL SIM time vs #phi (+Z, Single(topo1D)/First(topo2D) disk);#phi_{SIM} [rad];T_{SIM} [ns]",
                        50,
                        -3.15,
                        3.15,
                        0.,
                        100.);
  meHitTvsPhi_[3] = ibook.bookProfile("EtlHitTvsPhiZposD2",
                                      "ETL SIM time vs #phi (+Z, Second disk);#phi_{SIM} [rad];T_{SIM} [ns]",
                                      50,
                                      -3.15,
                                      3.15,
                                      0.,
                                      100.);

  meHitTvsEta_[0] =
      ibook.bookProfile("EtlHitTvsEtaZnegD1",
                        "ETL SIM time vs #eta (-Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM};T_{SIM} [ns]",
                        50,
                        -3.2,
                        -1.56,
                        0.,
                        100.);
  meHitTvsEta_[1] = ibook.bookProfile(
      "EtlHitTvsEtaZnegD2", "ETL SIM time vs #eta (-Z, Second disk);#eta_{SIM};T_{SIM} [ns]", 50, -3.2, -1.56, 0., 100.);
  meHitTvsEta_[2] =
      ibook.bookProfile("EtlHitTvsEtaZposD1",
                        "ETL SIM time vs #eta (+Z, Single(topo1D)/First(topo2D) disk);#eta_{SIM};T_{SIM} [ns]",
                        50,
                        1.56,
                        3.2,
                        0.,
                        100.);
  meHitTvsEta_[3] = ibook.bookProfile(
      "EtlHitTvsEtaZposD2", "ETL SIM time vs #eta (+Z, Second disk);#eta_{SIM};T_{SIM} [ns]", 50, 1.56, 3.2, 0., 100.);

  meNSimHitsPerEnteringTrackD1_ =
      ibook.book2D("NSimHitsPerEnteringTrackD1",
                   "ETL SIM hits per entering track in D1;N_{SIM hits in D1 per originalTrackId};p_{T}^{SimTrack at production} [GeV]",
                   15,
                   -0.5,
                   14.5,
                   25,
                   0.,
                   10.);

  meNSimHitsPerEnteringTrackD2_ =
      ibook.book2D("NSimHitsPerEnteringTrackD2",
                   "ETL SIM hits per entering track in D2;N_{SIM hits in D2 per originalTrackId};p_{T}^{SimTrack at production} [GeV]",
                   15,
                   -0.5,
                   14.5,
                   25,
                   0.,
                   10.);

  meNSimHitsFace2VsFace1D1_ =
      ibook.book2D("NSimHitsFace2VsFace1D1",
                   "ETL SIM hits per entering track in D1;N_{SIM hits on front face};N_{SIM hits on back face}",
                   10,
                   -0.5,
                   9.5,
                   10,
                   -0.5,
                   9.5);

  meNSimHitsFace2VsFace1D2_ =
      ibook.book2D("NSimHitsFace2VsFace1D2",
                   "ETL SIM hits per entering track in D2;N_{SIM hits on front face};N_{SIM hits on back face}",
                   10,
                   -0.5,
                   9.5,
                   10,
                   -0.5,
                   9.5);

  meNSimHitsFace2VsFace1D1NoOffset4_ =
      ibook.book2D("NSimHitsFace2VsFace1D1NoOffset4",
                   "ETL SIM hits per entering track in D1 excluding offsetTrackId == 4;N_{SIM hits on front face};N_{SIM hits on back face}",
                   10,
                   -0.5,
                   9.5,
                   10,
                   -0.5,
                   9.5);

  meNSimHitsFace2VsFace1D2NoOffset4_ =
      ibook.book2D("NSimHitsFace2VsFace1D2NoOffset4",
                   "ETL SIM hits per entering track in D2 excluding offsetTrackId == 4;N_{SIM hits on front face};N_{SIM hits on back face}",
                   10,
                   -0.5,
                   9.5,
                   10,
                   -0.5,
                   9.5);

  meNBackScatterVsFrontHitsD1_ =
      ibook.book2D("NBackScatterVsFrontHitsD1",
                   "ETL SIM hits per entering track in D1;N_{hits with offsetTrackId == 0};N_{hits with offsetTrackId == 4}",
                   15,
                   -0.5,
                   14.5,
                   15,
                   -0.5,
                   14.5);

  meNBackScatterVsFrontHitsD2_ =
      ibook.book2D("NBackScatterVsFrontHitsD2",
                   "ETL SIM hits per entering track in D2;N_{hits with offsetTrackId == 0};N_{hits with offsetTrackId == 4}",
                   15,
                   -0.5,
                   14.5,
                   15,
                   -0.5,
                   14.5);

  meSpaceDispersionXYD1_ =
      ibook.book1D("SpaceDispersionXYD1",
                   "ETL SIM hit space dispersion in D1;max pairwise #Delta r_{xy} [cm];Entries",
                   100,
                   0.,
                   200.);

  meSpaceDispersionXYD2_ =
      ibook.book1D("SpaceDispersionXYD2",
                   "ETL SIM hit space dispersion in D2;max pairwise #Delta r_{xy} [cm];Entries",
                   100,
                   0.,
                   200.);

  meTimeDispersionD1_ =
      ibook.book1D("TimeDispersionD1",
                   "ETL SIM hit time dispersion in D1;max(ToF)-min(ToF) [ns];Entries",
                   100,
                   0.,
                   500.);

  meTimeDispersionD2_ =
      ibook.book1D("TimeDispersionD2",
                   "ETL SIM hit time dispersion in D2;max(ToF)-min(ToF) [ns];Entries",
                   100,
                   0.,
                   500.);

  meTimeDispersionLatestOffset4D1_ =
      ibook.book1D("TimeDispersionLatestOffset4D1",
                   "ETL SIM hit time dispersion in D1 using only hits with offsetTrackId == 0;max(ToF)-min(ToF) [ns];Entries",
                   100,
                   0.,
                   500.);

  meTimeDispersionLatestOffset4D2_ =
      ibook.book1D("TimeDispersionLatestOffset4D2",
                   "ETL SIM hit time dispersion in D2 using only hits with offsetTrackId == 0;max(ToF)-min(ToF) [ns];Entries",
                   100,
                   0.,
                   500.);

  meNHitVsSpaceDispersionXYD1_ =
      ibook.book2D("NHitVsSpaceDispersionXYD1",
                   "ETL SIM hit space dispersion vs hit count in D1;N_{SIM hits in D1 per originalTrackId};max pairwise #Delta r_{xy} [cm]",
                   15,
                   -0.5,
                   14.5,
                   100,
                   0.,
                   200.);

  meNHitVsSpaceDispersionXYD2_ =
      ibook.book2D("NHitVsSpaceDispersionXYD2",
                   "ETL SIM hit space dispersion vs hit count in D2;N_{SIM hits in D2 per originalTrackId};max pairwise #Delta r_{xy} [cm]",
                   15,
                   -0.5,
                   14.5,
                   100,
                   0.,
                   200.);

  meNHitVsTimeDispersionD1_ =
      ibook.book2D("NHitVsTimeDispersionD1",
                   "ETL SIM hit time dispersion vs hit count in D1;N_{SIM hits in D1 per originalTrackId};max(ToF)-min(ToF) [ns]",
                   15,
                   -0.5,
                   14.5,
                   100,
                   0.,
                   500.);

  meNHitVsTimeDispersionD2_ =
      ibook.book2D("NHitVsTimeDispersionD2",
                   "ETL SIM hit time dispersion vs hit count in D2;N_{SIM hits in D2 per originalTrackId};max(ToF)-min(ToF) [ns]",
                   15,
                   -0.5,
                   14.5,
                   100,
                   0.,
                   500.);

  if (optionalPlots_) {
    meHitThetaEntryD1_[0] =
        ibook.book1D("HitThetaEntryD1_eta1", "ETL SIM hits D1 theta at entry, 1.5 < |eta| <= 2.1", 60, 0., 180.);
    meHitThetaEntryD1_[1] =
        ibook.book1D("HitThetaEntryD1_eta2", "ETL SIM hits D1 theta at entry, 2.1 < |eta| <= 2.5", 60, 0., 180.);
    meHitThetaEntryD1_[2] =
        ibook.book1D("HitThetaEntryD1_eta3", "ETL SIM hits D1 theta at entry, 2.5 < |eta| <= 3.0", 60, 0., 180.);

    meHitThetaEntryD2_[0] =
        ibook.book1D("HitThetaEntryD2_eta1", "ETL SIM hits D2 theta at entry, 1.5 < |eta| <= 2.1", 60, 0., 180.);
    meHitThetaEntryD2_[1] =
        ibook.book1D("HitThetaEntryD2_eta2", "ETL SIM hits D2 theta at entry, 2.1 < |eta| <= 2.5", 60, 0., 180.);
    meHitThetaEntryD2_[2] =
        ibook.book1D("HitThetaEntryD2_eta3", "ETL SIM hits D2 theta at entry, 2.5 < |eta| <= 3.0", 60, 0., 180.);
  }
}

void EtlSimHitsValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/ETL/SimHits");
  desc.add<edm::InputTag>("inputTag", edm::InputTag("mix", "g4SimHitsFastTimerHitsEndcap"));
  desc.add<edm::InputTag>("simTrackTag", edm::InputTag("g4SimHits"));
  desc.add<double>("hitMinimumEnergy2Dis", 0.001);
  desc.add<bool>("optionalPlots", false);

  descriptions.add("etlSimHitsValid", desc);
}

DEFINE_FWK_MODULE(EtlSimHitsValidation);