// Make oscillated predictions for varying parameters
// cafe VaryingParams.C
#include <math.h>
#include "CAFAna/Core/SpectrumLoader.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/Binning.h"
#include "CAFAna/Core/Var.h"
#include "CAFAna/Cuts/TruthCuts.h"
#include "StandardRecord/StandardRecord.h"
#include "TCanvas.h"
#include "TH1.h"
#include "CAFAna/Prediction/PredictionNoExtrap.h"
#include "CAFAna/Analysis/Calcs.h"
#include "OscLib/func/OscCalculatorPMNSOpt.h"
#include "CAFAna/Experiment/SingleSampleExperiment.h"
#include "CAFAna/Analysis/Fit.h"
#include "CAFAna/Vars/FitVars.h"
using namespace ana;
void VaryingParams()
{
  // Repeat all of demo1.C to get us our Prediction object
  const std::string fnameNonSwap = "/vols/dune/awaldron/data/ana_inputs/FD_FHC_nonswap.root";
  const std::string fnameNueSwap = "/vols/dune/awaldron/data/ana_inputs/FD_FHC_nueswap.root";
  const std::string fnameTauSwap = "/vols/dune/awaldron/data/ana_inputs/FD_FHC_tauswap.root";
  SpectrumLoader loaderNonSwap(fnameNonSwap);
  SpectrumLoader loaderNueSwap(fnameNueSwap);
  SpectrumLoader loaderTauSwap(fnameTauSwap);
  const Var kRecoEnergy = SIMPLEVAR(dune.Ev_reco_nue);
  const Binning binsEnergy = Binning::Simple(40, 0, 10);
  const HistAxis axEnergy("Reco energy (GeV)", binsEnergy, kRecoEnergy);
  const double pot = 3.5 * 1.47e21 * 40/1.13;
  const Cut kPassesCVN = SIMPLEVAR(dune.cvnnue) > .5;
  PredictionNoExtrap pred(loaderNonSwap, loaderNueSwap, loaderTauSwap, axEnergy, kPassesCVN);
  loaderNonSwap.Go();
  loaderNueSwap.Go();
  loaderTauSwap.Go();
  // We make the oscillation calculator "adjustable" so the fitter can
  // manipulate it.
  TObjArray Hlist(0);
  TFile f("histograms102400.root","recreate");
  char name[10];

  for(double i = 0.0874; i <= 0.1; i = i + (0.0000394)) {
    for(double j = -1; j <= 1; j = j + 0.00625 {
      osc::IOscCalculatorAdjustable* calc = DefaultOscCalc();
      double th13 = (asin (sqrt(i)))/2;
      double dcp = (asin (j));
      calc->SetTh13(th13);
      calc->SetdCP(dcp);
      //std::cout<<calc->GetTh13();
      //std::cout<<calc->GetdCP();
      cout << i << " for sin2(2Theta13)\n";
      cout << calc->GetTh13() << " for Theta13\n";
      cout << calc->GetdCP() << " for DeltaCP\n";
      const Spectrum sOsc = pred.Predict(calc);
      std::string hist_i = std::to_string(i);
      std::string hist_j = std::to_string(j);
      std::string hist_name = "figures/"+hist_i+"_"+hist_j+".png";
      const char *out = hist_name.c_str();
      std::string canvas_name = "c"+hist_i+"_"+hist_j;
      const char *canvas = canvas_name.c_str();
      TCanvas* c1 = new TCanvas(canvas);
      // sOsc.ToTH1(pot, kRed)->Draw("hist");
      TH1* h = sOsc.ToTH1(pot,kRed);
      h->SetName(canvas);
      h->Draw("hist");
      h->Write();
      // c1->SaveAs(out);
    }
  }

    f.Close();
    return;
}
