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
    int face = -1;
    int offsetTrackId = -999;
  };

  std::array<int, 2> nSimHitsPerDisk{{0, 0}};
  std::array<std::array<int, 2>, 2> nSimHitsPerDiskFace{{{{0, 0}}, {{0, 0}}}};
  std::array<std::vector<HitInfo>, 2> hitsPerDisk;

  float trackPtAtProduction = -1.f;
  bool hasTrackPtAtProduction = false;

  void setTrackPtAtProduction(float pt) {
    trackPtAtProduction = pt;
    hasTrackPtAtProduction = true;
  }

  void addHit(int disc, int face, float x, float y, float z, float tof, int offsetTrackId) {
    if (disc < 1 || disc > 2)
      return;

    const int diskIndex = disc - 1;
    ++nSimHitsPerDisk[diskIndex];

    if (face == 0 || face == 1) {
      ++nSimHitsPerDiskFace[diskIndex][face];
    }

    hitsPerDisk[diskIndex].push_back({x, y, z, tof, face, offsetTrackId});
  }
};

struct DispersionResult {
  float maxPairwiseXY = 0.f;
  float maxPairwise3D = 0.f;
  float rmsXY = 0.f;
  float timeSpread = 0.f;
  int latestOffsetTrackId = -999;
};

static DispersionResult computeDispersion(const std::vector<EnteringTrackDiskSummary::HitInfo>& hits) {
  DispersionResult result;

  const size_t nHits = hits.size();
  if (nHits == 0) {
    return result;
  }

  float minTof = std::numeric_limits<float>::max();
  float maxTof = -std::numeric_limits<float>::max();

  float meanX = 0.f;
  float meanY = 0.f;

  for (const auto& hit : hits) {
    meanX += hit.x;
    meanY += hit.y;

    if (hit.tof < minTof)
      minTof = hit.tof;
    if (hit.tof > maxTof) {
      maxTof = hit.tof;
      result.latestOffsetTrackId = hit.offsetTrackId;
    }
  }

  meanX /= static_cast<float>(nHits);
  meanY /= static_cast<float>(nHits);

  float sumR2 = 0.f;
  for (const auto& hit : hits) {
    const float dx = hit.x - meanX;
    const float dy = hit.y - meanY;
    sumR2 += dx * dx + dy * dy;
  }

  result.rmsXY = std::sqrt(sumR2 / static_cast<float>(nHits));
  result.timeSpread = maxTof - minTof;

  for (size_t i = 0; i < nHits; ++i) {
    for (size_t j = i + 1; j < nHits; ++j) {
      const float dx = hits[i].x - hits[j].x;
      const float dy = hits[i].y - hits[j].y;
      const float dz = hits[i].z - hits[j].z;

      const float distXY = std::sqrt(dx * dx + dy * dy);
      const float dist3D = std::sqrt(dx * dx + dy * dy + dz * dz);

      if (distXY > result.maxPairwiseXY)
        result.maxPairwiseXY = distXY;

      if (dist3D > result.maxPairwise3D)
        result.maxPairwise3D = dist3D;
    }
  }

  return result;
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

  MonitorElement* meSpaceDispersionXYD1_ = nullptr;
  MonitorElement* meSpaceDispersionXYD2_ = nullptr;
  MonitorElement* meSpaceDispersionRMSD1_ = nullptr;
  MonitorElement* meSpaceDispersionRMSD2_ = nullptr;
  MonitorElement* meTimeDispersionD1_ = nullptr;
  MonitorElement* meTimeDispersionD2_ = nullptr;
  MonitorElement* meTimeDispersionLatestOffset4D1_ = nullptr;
  MonitorElement* meTimeDispersionLatestOffset4D2_ = nullptr;

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

    auto& enteringTrackSummary = enteringTracks[simHit.originalTrackId()];
    enteringTrackSummary.addHit(id.nDisc(),
                                id.discSide(),
                                globalPointForThisHit.x(),
                                globalPointForThisHit.y(),
                                globalPointForThisHit.z(),
                                simHit.tof(),
                                simHit.offsetTrackId());

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

    if (summary.nSimHitsPerDisk[0] > 0 && summary.hasTrackPtAtProduction) {
      meNSimHitsPerEnteringTrackD1_->Fill(summary.nSimHitsPerDisk[0], summary.trackPtAtProduction);
    }

    if (summary.nSimHitsPerDisk[1] > 0 && summary.hasTrackPtAtProduction) {
      meNSimHitsPerEnteringTrackD2_->Fill(summary.nSimHitsPerDisk[1], summary.trackPtAtProduction);
    }

    if (summary.nSimHitsPerDisk[0] > 0) {
      meNSimHitsFace2VsFace1D1_->Fill(summary.nSimHitsPerDiskFace[0][0], summary.nSimHitsPerDiskFace[0][1]);

      const auto dispersionD1 = computeDispersion(summary.hitsPerDisk[0]);
      meSpaceDispersionXYD1_->Fill(dispersionD1.maxPairwiseXY);
      meSpaceDispersionRMSD1_->Fill(dispersionD1.rmsXY);
      meTimeDispersionD1_->Fill(dispersionD1.timeSpread);
      if (dispersionD1.latestOffsetTrackId == 0) {
        meTimeDispersionLatestOffset4D1_->Fill(dispersionD1.timeSpread);
      }
    }

    if (summary.nSimHitsPerDisk[1] > 0) {
      meNSimHitsFace2VsFace1D2_->Fill(summary.nSimHitsPerDiskFace[1][0], summary.nSimHitsPerDiskFace[1][1]);

      const auto dispersionD2 = computeDispersion(summary.hitsPerDisk[1]);
      meSpaceDispersionXYD2_->Fill(dispersionD2.maxPairwiseXY);
      meSpaceDispersionRMSD2_->Fill(dispersionD2.rmsXY);
      meTimeDispersionD2_->Fill(dispersionD2.timeSpread);
      if (dispersionD2.latestOffsetTrackId == 0) {
        meTimeDispersionLatestOffset4D2_->Fill(dispersionD2.timeSpread);
      }
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
                   100,
                   0.,
                   20.);

  meNSimHitsPerEnteringTrackD2_ =
      ibook.book2D("NSimHitsPerEnteringTrackD2",
                   "ETL SIM hits per entering track in D2;N_{SIM hits in D2 per originalTrackId};p_{T}^{SimTrack at production} [GeV]",
                   15,
                   -0.5,
                   14.5,
                   100,
                   0.,
                   20.);

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

  meSpaceDispersionRMSD1_ =
      ibook.book1D("SpaceDispersionRMSD1",
                   "ETL SIM hit RMS space dispersion in D1;RMS #Delta r_{xy} [cm];Entries",
                   100,
                   0.,
                   100.);

  meSpaceDispersionRMSD2_ =
      ibook.book1D("SpaceDispersionRMSD2",
                   "ETL SIM hit RMS space dispersion in D2;RMS #Delta r_{xy} [cm];Entries",
                   100,
                   0.,
                   100.);

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
                   "ETL SIM hit time dispersion in D1, latest hit offsetTrackId == 0;max(ToF)-min(ToF) [ns];Entries",
                   100,
                   0.,
                   500.);

  meTimeDispersionLatestOffset4D2_ =
      ibook.book1D("TimeDispersionLatestOffset4D2",
                   "ETL SIM hit time dispersion in D2, latest hit offsetTrackId == 0;max(ToF)-min(ToF) [ns];Entries",
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